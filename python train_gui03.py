import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import os
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, random_split
from llama_cpp import Llama
from llama_cpp.llama import Tokenizer

class LoRA_model(torch.nn.Module):
    def __init__(self, original_model, vocab_size=30522):
        super().__init__()
        self.original_model = original_model
        self.lora_input = torch.nn.Linear(original_model.dim, 1)  # 假设Llama模型的隐藏维度为 'dim'
        self.lora_output = torch.nn.Linear(1, vocab_size)
        
    def forward(self, input_ids):
        # 假设原始模型的输出为 logits
        outputs = self.original_model.eval().generate(input_ids=input_ids)
        hidden_state = outputs[:, -1, :]  # 使用最后一个标记的隐藏状态
        logit = self.lora_input(hidden_state)
        logit = logit.unsqueeze(1)  # 扩展维度
        predictions = self.lora_output(logit)
        return predictions

class TrainApp:
    def __init__(self):
        self.model_name = None
        self.learning_rate = 0.0001
        self.num_epochs = 10
        self.batch_size = 32
        self.train_dataset = None
        self.progress_bar = None
        self.original_model = None
    
    def initialize_model(self):
        # 假设预训练模型存储在 pretrained_models 文件夹中
        model_dir = "pretrained_models"
        os.makedirs(model_dir, exist_ok=True)  # 确保文件夹存在

        # 动态生成模型路径
        if self.model_name is None:
            messagebox.showerror("错误", "请先选择预训练模型！")
            return None

        # 支持 .pth 和 .gguf 文件
        model_filename = f"{self.model_name}.pth"
        gguf_model_filename = f"{self.model_name}.gguf"

        if os.path.exists(os.path.join(model_dir, gguf_model_filename)):
            # 转换 .gguf 文件为 PyTorch 模型
            from llama_cpp import Llama
            model_path = os.path.join(model_dir, gguf_model_filename)
            self.original_model = Llama(model_path)
            print(f"成功加载 GGUF 模型: {model_path}")
        else:
            # 加载 .pth 模型
            model_path = os.path.join(model_dir, model_filename)
            if os.path.exists(model_path):
                self.original_model = torch.load(model_path)
                print(f"成功加载预训练模型: {model_path}")
            else:
                print(f"预训练模型文件 {model_path} 不存在，请先下载模型！")
                return None

        # 设置模型参数为可训练（仅对 PyTorch 模型生效）
        if isinstance(self.original_model, torch.nn.Module):
            for param in self.original_model.parameters():
                param.requires_grad = True

        return self.original_model
    
    def get_available_models(self):
        model_dir = "pretrained_models"  # 预训练模型文件夹

        # 如果文件夹不存在，返回空列表
        if not os.path.exists(model_dir):
            return []

        # 获取所有支持的文件，并提取文件名作为模型名称
        model_files = [f for f in os.listdir(model_dir) if f.endswith((".pth", ".gguf"))]

        # 去除扩展名
        model_names = [os.path.splitext(f)[0] for f in model_files]

        return model_names
    
    def browse_data(self):
        filename = filedialog.askopenfilename(title="打开文件", defaultextension=".txt")
        if filename:
            data_path = os.path.abspath(filename)
            with open(data_path, "r", encoding="utf-8") as f:
                all_text = f.read()
            texts = all_text.split("\n\n")
            dataset = []
            for i in range(0, len(texts), 2):
                text1 = texts[i].strip()
                text2 = texts[i+1].strip() if i+1 < len(texts) else ""
                dataset.append((text1, text2))
            self.train_dataset = dataset  # 将数据存储到实例变量
        else:
            messagebox.showwarning("警告", "未选择文件！")
    
    def train_model(self):
        if not self.initialize_model():
            messagebox.showerror("错误", "无法加载预训练模型！")
            return
        
        # 数据集必须存在
        if not self.train_dataset:
            messagebox.showerror("错误", "未加载训练数据！")
            return
        
        # 定义数据集类
        class TextDataset(Dataset):
            def __init__(self, data):
                self.data = data
                self.tokenizer = self.get_tokenizer()  # 添加分词器
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                text1, text2 = self.data[idx]
                # 使用分词器将文本转换为张量
                tokens1 = self.tokenizer.encode(text1, add_bos=True, add_eos=True)
                tokens2 = self.tokenizer.encode(text2, add_bos=True, add_eos=True)
                input_ids = tokens1.ids
                targets = tokens2.ids
                return {
                    "input_ids": torch.tensor(input_ids),  # 假设 text1 已编码为张量
                    "targets": torch.tensor(targets)       # 假设 text2 是标签
                }
            
            def get_tokenizer(self):
                # 返回分词器实例
                return Tokenizer(model_path=os.path.join("pretrained_models", f"{self.model_name}.tokenizer"))  # 假设分词器与模型同名

        train_dataset = TextDataset(self.train_dataset)
        val_split = 0.1
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # 初始化 LoRA 模型
        self.model_with_lora = LoRA_model(self.original_model)
        
        # 定义优化器和损失函数
        optimizer = torch.optim.AdamW(
            self.model_with_lora.parameters(),
            lr=self.learning_rate
        )
        criterion = CrossEntropyLoss()
        
        # 训练过程
        self.progress_bar.start()
        for epoch in range(self.num_epochs):
            self.model_with_lora.train()
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model_with_lora(batch["input_ids"])
                loss = criterion(outputs, batch["targets"].unsqueeze(1))  # 确保标签与预测的形状一致
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch["input_ids"].size(0)
            
            avg_loss = running_loss / len(train_loader.dataset)
            
            # 验证过程
            self.model_with_lora.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = self.model_with_lora(batch["input_ids"])
                    val_loss = criterion(outputs, batch["targets"].unsqueeze(1))
                    val_running_loss += val_loss.item() * batch["input_ids"].size(0)
            
            avg_val_loss = val_running_loss / len(val_loader.dataset)
            
            # 更新进度条
            progress = int(100.0 * (epoch + 1) / self.num_epochs)
            self.progress_bar["value"] = progress
            self.progress_bar.update()
            
            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        self.progress_bar.stop()
        self.progress_bar["value"] = 100
        
        # 保存模型
        model_dir = "trained_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.model_name}_finetuned.pt")  # 保存为 PyTorch 格式
        torch.save(self.model_with_lora.state_dict(), model_path)
        messagebox.showinfo("信息", f"训练完成，模型已保存在{model_path}")
    
    def show_ui(self, master):
        main_frame = ttk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 模型选择
        model_label = ttk.Label(main_frame, text="预训练模型")
        model_label.pack(anchor=tk.W, pady=5)
        
        model_names = self.get_available_models()
        if not model_names:
            messagebox.showwarning("警告", "未找到预训练模型，请检查 pretrained_models 文件夹！")
        else:
            self.model_combobox = ttk.Combobox(main_frame, values=model_names)
            self.model_combobox.set(model_names[0] if model_names else "")
            self.model_combobox.pack(fill=tk.X, padx=5, pady=5)
            self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_selected)
        
        # 数据上传
        data_label = ttk.Label(main_frame, text="训练数据文件")
        data_label.pack(anchor=tk.W, pady=5)
        
        self.data_button = ttk.Button(main_frame, text="选择文件", command=self.browse_data)
        self.data_button.pack(fill=tk.X, padx=5, pady=5)
        
        # 训练参数设置
        param_frame = ttk.LabelFrame(main_frame, text="训练参数")
        param_frame.pack(fill=tk.X, pady=5)
        
        params = [
            ("批次大小", self.batch_size),
            ("学习率", self.learning_rate),
            ("训练轮数", self.num_epochs)
        ]
        
        self.param_entries = {}
        for i, (label, value) in enumerate(params):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, padx=5, pady=5)
            entry = ttk.Entry(param_frame)
            entry.insert(0, str(value))
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.param_entries[label] = entry
        
        # 进度条
        self.progress_bar = ttk.Progressbar(main_frame, length=300, mode='determinate')
        self.progress_bar.pack(pady=10)
        
        # 训练开始按钮
        self.train_button = ttk.Button(main_frame, text="开始训练", command=self.train_model)
        self.train_button.pack(pady=10)
        
        # 关闭按钮
        close_button = ttk.Button(main_frame, text="关闭", command=lambda: master.destroy())
        close_button.pack(pady=5, padx=5, side=tk.BOTTOM, anchor=tk.SE)
        
    def on_model_selected(self, event):
        self.model_name = self.model_combobox.get()
    
    def update_train_params(self):
        # 从输入框中读取参数并更新实例变量
        self.batch_size = int(self.param_entries["批次大小"].get())
        self.learning_rate = float(self.param_entries["学习率"].get())
        self.num_epochs = int(self.param_entries["训练轮数"].get())

def main():
    root = tk.Tk()
    root.title("模型微调工具")
    app = TrainApp()
    app.show_ui(root)
    root.mainloop()

if __name__ == "__main__":
    main()
    print{"hello world"}
    