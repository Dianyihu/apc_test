# endcoding: utf-8

'''
Created by
@author: Dianyi Hu
@date: 2024/3/23 
@time: 01:20
'''

import asyncio
import time
import multiprocessing

# ----------------------------Break----------------------------

# lock = threading.Lock()


class FP_Machine:
    def __init__(self):
        pass

    def load_model(self, model_path):
        pass

    def update_model(self, time_point):
        pass

    def loss_fun(self):
        pass

    def opti_p(self):
        pass


def resolve_json(msg):
    fp_equip_id, apc_parameters, wafer_info = 'fake_machine', {}, {}
    return fp_equip_id, apc_parameters, wafer_info

async def work_dispatcher(msg):
    # fp_equip_id, head, wafer_id = receive.values
    fp_equip_id, apc_parameters, wafer_info = resolve_json(msg)
    print(fp_equip_id, apc_parameters, wafer_info)

    if fp_equip_id in globals():
        fp_equip_id


class APC_Sever(multiprocessing.Process):
    def __init__(self):
        super().__init__()

    async def handle_echo(self, reader, writer):
        data = await reader.read(100)
        message = data.decode()

        addr = writer.get_extra_info('peername')
        print("接收来自客户端：", addr, '的数据是：', message)

        await work_dispatcher(message)

        # 写入数据
        writer.write(data)
        await writer.drain()
        print("给客户端发送的数据是：", message)

        print(addr, "客户端进行了连接关闭")
        writer.close()
        await writer.wait_closed()

    async def start_server(self):
        server = await asyncio.start_server(
            self.handle_echo, '0.0.0.0', 8000)

        addr = server.sockets[0].getsockname()

        print(f'服务运行: {addr}')

        async with server:
            await server.serve_forever()

    def run(self):
        asyncio.run(self.start_server())


def testfun2():
    for i in range(10):
        i += 1
        print(i)
        time.sleep(1)


if __name__ == '__main__':
    p1 = APC_Sever()

    p1.start()
    p1.join()

    print('Hello here!')
