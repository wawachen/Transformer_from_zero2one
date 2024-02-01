from torch.utils.data import Dataset
import json

class C2E_translate(Dataset):
    def __init__(self,file_path, limit_num = None) -> None:
        super().__init__()
        self.data = self.load_data(file_path, limit=limit_num)

    def load_data(self, file_path, limit=None):
        data = {}
        with open(file_path,'rt',encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if limit is not None:
                    if idx >=limit:
                        break
                sample = json.loads(line.strip())
                data[idx] = sample
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
if __name__ == "__main__":
    path = "/home/wawa/pytorch-transformer/my_reproduce/translation2019zh/translation2019zh_train.json"
    data = C2E_translate(path)

    print(len(data))
    print(next(iter(data)))
