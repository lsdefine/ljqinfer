from collections import deque
class BatchScheduler:
    def __init__(self): self.queue=deque()
    def submit(self,item): self.queue.append(item)
    def pop(self): return self.queue.popleft()
