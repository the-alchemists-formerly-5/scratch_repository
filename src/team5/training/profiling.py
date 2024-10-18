from torch.profiler import profile, record_function, ProfilerActivity
from transformers import TrainerCallback

class ProfilingCallback(TrainerCallback):
    def __init__(self, device, train_dataset, peft_model, n_steps=10):
        self.device = device
        self.train_dataset = train_dataset
        self.peft_model = peft_model
        self.n_steps = n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.n_steps == 0:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         profile_memory=True, record_shapes=True) as prof:
                with record_function("model_inference"):
                    # Run a forward pass
                    example = self.train_dataset[0]
                    # Each field in the example is a tensor, so we need to add a batch dimension to the front of each
                    example = {k: v.unsqueeze(0).to(self.device) for k, v in example.items()}
                    self.peft_model(**example)
            
            print(f"Step {state.global_step}")
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))

def create_profiling_callback(device, train_dataset, peft_model, n_steps=10):
    return ProfilingCallback(device, train_dataset, peft_model, n_steps)