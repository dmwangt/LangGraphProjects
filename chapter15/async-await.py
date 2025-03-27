import asyncio

shared_resource = []


async def add_to_resource(task_name, sleep=1):
    print(f"{task_name} adding to resource..., sleep={sleep}")
    await asyncio.sleep(sleep)  # Simulate I/O
    shared_resource.append(task_name)
    print(f"{task_name} added to resource: {shared_resource}")


async def main():
    await asyncio.gather(add_to_resource("Task 1", sleep=2), add_to_resource("Task 2"))


asyncio.run(main())
