"""不加载 GPU 模型，验证真实会话协程的背压、断连及输入校验。"""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / 'app_ws-multi.py'


def load_functions(*names):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            functions.append(node)
    if {node.name for node in functions} != set(names):
        raise AssertionError('应用入口发生变化，请更新测试加载器')
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in {
                     'JOB_RETENTION_SECONDS', 'JOB_CLEANUP_MAX_FAILURES',
                     'WS_SEND_TIMEOUT_SECONDS', 'WS_CLOSE_TIMEOUT_SECONDS',
                 } for target in node.targets)]
    namespace = dict(globals(), WebSocket=object, WebSocketDisconnect=ConnectionError,
                     ThreadPoolExecutor=ThreadPoolExecutor, math=math)
    # Only execute selected functions from the checked-in application, never user input.
    exec(compile(ast.Module(body=constants + functions, type_ignores=[]), str(SOURCE), 'exec'), namespace)  # pylint: disable=exec-used
    return namespace


class FrameRateTests(unittest.TestCase):
    def test_configuration(self):
        namespace = load_functions('configuration')
        namespace['get_pool'] = Mock(return_value=Mock(engines={'test': {}}))
        config = dict(engine='test', width=640, height=480)
        configure = namespace['configuration']
        self.assertEqual(configure(config)['fps'], 60)
        for value in (10, 15, 20, 24, 25, 30, 40, 45, 50, 60, '30', 30.0):
            with self.subTest(value=value):
                self.assertEqual(configure(dict(config, fps=value))['fps'], int(value))
        for value in ('abc', '', '30.9', 30.9, None, True, False, [], {},
                      float('inf'), float('-inf'), float('nan'), 0, 31):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, '^无效帧率$'):
                    configure(dict(config, fps=value))


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.namespace = load_functions('cleanup_jobs')
        self.job = dict(status='done', updated=0, finished_at=0, directory='/unused', readers=0)
        self.namespace.update(jobs={'job': self.job}, jobs_lock=threading.Lock())
        self.cleanup = self.namespace['cleanup_jobs']

    def test_failures_back_off_and_stop_but_keep_record(self):
        with patch.object(shutil, 'rmtree', side_effect=PermissionError('denied')) as remove, \
                patch.object(time, 'time', return_value=1000) as clock, patch('builtins.print'):
            for attempt in range(1, 6):
                self.cleanup()
                self.assertEqual(remove.call_count, attempt)
                self.assertEqual(self.job['cleanup_failures'], attempt)
                if attempt < 5:
                    self.assertEqual(self.job['cleanup_retry_at'], clock.return_value + 10 * 2 ** (attempt - 1))
                    self.cleanup()
                    self.assertEqual(remove.call_count, attempt, '退避期间不重复删除')
                    clock.return_value = self.job['cleanup_retry_at']
            self.assertTrue(self.job['cleanup_abandoned'])
            self.assertEqual(self.job['cleanup_error'], 'denied')
            clock.return_value += 10000
            self.cleanup()
            self.assertEqual(remove.call_count, 5)
            self.assertIn('job', self.namespace['jobs'])

    def test_successful_retry_removes_directory_and_record(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'job'
            directory.mkdir()
            (directory / 'result').write_bytes(b'result')
            self.job['directory'] = str(directory)
            with patch.object(time, 'time', return_value=1000) as clock:
                with patch.object(shutil, 'rmtree', side_effect=PermissionError('busy')), patch('builtins.print'):
                    self.cleanup()
                clock.return_value = self.job['cleanup_retry_at']
                self.cleanup()
            self.assertFalse(directory.exists())
            self.assertEqual(self.namespace['jobs'], {})

    def test_active_readers_and_workers_are_protected(self):
        with patch.object(shutil, 'rmtree') as remove, patch.object(time, 'time', return_value=1000):
            self.job['readers'] = 1
            self.cleanup()
            self.job.update(readers=0, future=Mock(done=Mock(return_value=False)))
            self.cleanup()
            remove.assert_not_called()


class FakeWebSocket:
    def __init__(self, block_init=False, block_close=False):
        self.incoming = asyncio.Queue()
        self.started = asyncio.Queue()
        self.sent = asyncio.Queue()
        self.gate = asyncio.Event()
        self.block_init = block_init
        self.block_close = block_close
        self.closed = False
        self.close_started = asyncio.Event()

    async def accept(self):
        pass

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, value):
        await self.send(value)

    async def send_bytes(self, packet):
        length = int.from_bytes(packet[:4], 'big')
        value = json.loads(packet[4:4 + length])
        value['_jpeg'] = packet[4 + length:]
        await self.send(value)

    async def send(self, value):
        self.started.put_nowait(value)
        if value['type'] != 'init' or self.block_init:
            await self.gate.wait()
        self.sent.put_nowait(value)

    async def close(self):
        self.closed = True
        self.close_started.set()
        if self.block_close:
            await asyncio.Event().wait()

    def frame(self, index, policy):
        self.incoming.put_nowait({'text': json.dumps(dict(type='frame', id=index, revision=1,
                                                        queue_policy=policy)), 'type': 'websocket.receive'})
        self.incoming.put_nowait({'bytes': b'input', 'type': 'websocket.receive'})

    def disconnect(self):
        self.incoming.put_nowait({'type': 'websocket.disconnect'})


class RealtimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.namespace = load_functions('realtime')
        self.processed = asyncio.Queue()
        self.jpeg = False
        loop = asyncio.get_running_loop()

        def process(raw, config):
            result = dict(type='result', id=config['id'])
            if self.jpeg:
                result['_jpeg'] = b'jpeg-' + str(config['id']).encode()
            loop.call_soon_threadsafe(self.processed.put_nowait, config['id'])
            return result

        self.namespace.update(pool=object(), RealtimePipeline=lambda pool: Mock(process=process))
        self.session = None

    async def asyncTearDown(self):
        if self.session is not None:
            self.session.cancel()
            await asyncio.gather(self.session, return_exceptions=True)

    async def start(self, socket):
        self.session = asyncio.create_task(self.namespace['realtime'](socket))
        self.assertEqual((await asyncio.wait_for(socket.started.get(), 2))['type'], 'init')
        if not socket.block_init:
            await asyncio.wait_for(socket.sent.get(), 2)

    async def frame(self, socket, index, policy):
        socket.frame(index, policy)
        self.assertEqual(await asyncio.wait_for(self.processed.get(), 2), index)

    async def exercise_slow_sender(self, policies, expected, jpeg=False):
        self.jpeg = jpeg
        socket = FakeWebSocket()
        await self.start(socket)
        await self.frame(socket, 0, policies[0])
        self.assertEqual((await asyncio.wait_for(socket.started.get(), 2))['id'], 0)
        for index, policy in enumerate(policies[1:], 1):
            await self.frame(socket, index, policy)
        # 等到推理协程等待下一帧或结果队列容量，避免释放发送门时最后一帧还在运行。
        async def inference_settled():
            while True:
                tasks = [task for task in asyncio.all_tasks()
                         if task.get_coro().__qualname__.endswith('realtime.<locals>.infer_fifo')]
                if tasks:
                    waiting = tasks[0].get_coro().cr_await
                    if waiting is not None and getattr(waiting, '__qualname__', '') in ('Queue.get', 'Queue.put'):
                        return
                await asyncio.sleep(.001)
        await asyncio.wait_for(inference_settled(), 2)
        socket.gate.set()
        results = [await asyncio.wait_for(socket.sent.get(), 2) for _ in expected]
        self.assertEqual([result['id'] for result in results], expected)
        for result in results:
            self.assertNotIn('_result_ready_at', result)
            self.assertAlmostEqual(result['server_total_ms'],
                                   result['queue_ms'] + result['server_process_ms'] + result['send_queue_ms'],
                                   delta=.03)
        if jpeg:
            for result in results:
                self.assertEqual(result['_jpeg'], b'jpeg-' + str(result['id']).encode())
        socket.disconnect()
        await asyncio.wait_for(self.session, 2)
        self.assertTrue(socket.closed)
        self.assertTrue(socket.sent.empty())
        return results

    async def test_fifo_infers_ahead_and_preserves_results(self):
        results = await self.exercise_slow_sender(['fifo'] * 6, list(range(6)), jpeg=True)
        self.assertTrue(all(result['replaced_results'] == 0 for result in results))

    async def test_latest_replaces_unsent_results(self):
        results = await self.exercise_slow_sender(['latest'] * 6, [0, 5])
        self.assertEqual(results[-1]['replaced_results'], 4)

    async def test_policy_change_preserves_queued_fifo_results(self):
        results = await self.exercise_slow_sender(['fifo', 'fifo', 'latest', 'latest', 'fifo', 'latest'],
                                                [0, 1, 4, 5])
        self.assertEqual(results[-1]['replaced_results'], 2)

    async def test_send_and_close_timeouts_reclaim_session(self):
        self.namespace.update(WS_SEND_TIMEOUT_SECONDS=.03, WS_CLOSE_TIMEOUT_SECONDS=.03)
        for block_init, jpeg in ((True, False), (False, False), (False, True)):
            with self.subTest(block_init=block_init, jpeg=jpeg), patch('builtins.print'):
                self.jpeg = jpeg
                socket = FakeWebSocket(block_init=block_init, block_close=True)
                await self.start(socket)
                if not block_init:
                    await self.frame(socket, 0, 'fifo')
                await asyncio.wait_for(self.session, 2)
                self.assertTrue(socket.closed)

    async def test_disconnect_cancels_blocked_sender(self):
        socket = FakeWebSocket()
        await self.start(socket)
        await self.frame(socket, 0, 'fifo')
        await asyncio.wait_for(socket.started.get(), 2)
        socket.disconnect()
        await asyncio.wait_for(self.session, 2)
        self.assertTrue(socket.closed)

    async def test_full_fifo_queues_still_time_out(self):
        self.namespace['WS_SEND_TIMEOUT_SECONDS'] = .1
        socket = FakeWebSocket()
        with patch('builtins.print'):
            await self.start(socket)
            for index in range(20):
                socket.frame(index, 'fifo')
            await asyncio.wait_for(self.session, 2)
        self.assertTrue(socket.closed)
        self.assertLess(self.processed.qsize(), 20, '背压应限制推理超前量')

    async def test_disconnect_closes_transport_before_waiting_for_inference(self):
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def process(raw, config):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(5):
                raise TimeoutError('测试未释放推理线程')
            return dict(type='result', id=config['id'])

        self.namespace['RealtimePipeline'] = lambda pool: Mock(process=process)
        socket = FakeWebSocket()
        try:
            await self.start(socket)
            socket.frame(0, 'fifo')
            await asyncio.wait_for(started.wait(), 2)
            socket.disconnect()
            await asyncio.wait_for(socket.close_started.wait(), 2)
            self.assertFalse(self.session.done(), '关闭传输后仍需等待正在执行的推理')
        finally:
            release.set()
        await asyncio.wait_for(self.session, 2)


if __name__ == '__main__':
    unittest.main()
