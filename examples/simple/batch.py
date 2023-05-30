import torch
from typing import List, Deque, Tuple, Hashable, Any
from energonai import BatchManager, SubmitEntry, TaskEntry


class BatchManagerForGeneration(BatchManager):
    def __init__(self, tokenizer, max_batch_size: int = 1) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.tokenizer = tokenizer


    @staticmethod
    def _make_batch_key(entry: SubmitEntry) -> tuple:
        data = entry.data
        return (data['top_k'], data['top_p'], data['temperature'])

    def make_batch(self, q: Deque[SubmitEntry]) -> Tuple[TaskEntry, dict]:
        entry = q.popleft()
        uids = [entry.uid]
        batch = [entry.data]
        while len(batch) < self.max_batch_size:
            if len(q) == 0:
                break
            e = q.popleft()
            batch.append(e.data)
            uids.append(e.uid)
        ret = self.tokenizer(batch, padding=True)
        inputs = {"input_ids":torch.tensor(ret["input_ids"], dtype=torch.long),
                  "attention_mask":torch.tensor(ret["attention_mask"], dtype=torch.long)}
        return TaskEntry(tuple(uids), inputs), {}

    def split_batch(self, task_entry: TaskEntry, **batch_info) -> List[Tuple[Hashable, Any]]:
        retval = []
        for uid, output in zip(task_entry.uids, task_entry.batch["logits"]):
            val, ind = output.max(dim=-1)
            output = self.tokenizer.decode(ind, skip_special_tokens=True)
            output += str(val.cpu().tolist())
            retval.append((uid, output))
        return retval
