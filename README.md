# chika

`chika` is a simple and easy config tool for hierarchical configurations.

## Requirements

* Python>=3.8

## Usage

Write typed configurations using `chika.config`, which is similar to `dataclass`.

```python
# main.py
import chika

@chika.config
class ModelConfig:
    name: str = chika.choices('resnet', 'densenet')

@chika.config
class DataConfig:
    # values that needs to be specified
    name: str = chika.required()

@chika.config
class OptimConfig:
    # 
    steps: List[int] = chika.sequence(100, 150)

@chika.config
class Config:
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig

    seed: int = chika.with_help(1, "random seed")
    use_amp: bool = False
    num_gpu: int = 1


@chika.main(config=Config)
def main(cfg: Config):
    model = ModelRegistry(cfg.model)
    ...

```

```yaml
# config.yaml
model:
  name: resnet
  ... 
```

### Expected Behavior

```commandline
python main.py --use_amp
# cfg.use_amp == True

python main.py --model config/densenet.yaml
# cfg.model.name == densenet

python main.py --model.name resnet
# cfg.model.name == resnet

python main.py --optim.decay_steps 100 150
# config.optim.decay_steps == [100, 150]
```

### Other APIs

```python
>>> cfg.show()
# model: name=resnet
#        zero_init=True
#        ...
# ...

>>> cfg.to_dict()
# {"model": {"name": "resnet", "zero_init": True, ...}, ...}

>>> cfg.from_file("config.yaml")
>>> cfg.from_file("config.json")

>>> chika.unique_path
# Path("outputs/202008200001-609938")
```