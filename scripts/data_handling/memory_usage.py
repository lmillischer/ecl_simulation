import psutil
import gc
import sys


def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / (1024 ** 2):.2f} MB, VMS: {mem_info.vms / (1024 ** 2):.2f} MB")


def get_largest_objects(limit=10):
    all_objects = gc.get_objects()
    object_sizes = [(obj, sys.getsizeof(obj)) for obj in all_objects]
    object_sizes.sort(key=lambda x: x[1], reverse=True)
    for obj, size in object_sizes[:limit]:
        print(f"Object type: {type(obj)}, Size: {size / 1024:.2f} KB")