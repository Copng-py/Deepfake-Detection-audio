# Deepfake-Detection-audio

Deepfake-Detection-audio is a research/engineering repo for training and evaluating audio deepfake detectors, with emphasis on generalization to unseen generators (i.e., test-time generators not present during training).

## Key ideas
- Train detectors on multiple known generators and bona fide audio.
- Evaluate generalization by holding out one or more generators during training and testing on those unseen generators.
- Provide reproducible scripts for preprocessing, training, and evaluation.

## Repo layout
```
├── config.py           # Configuration settings
├── main.py            # Main entry point
├── requirements.txt    # Python dependencies
└── network/           # Model implementations for EnvSDD competition
```

## Requirements
- Python 3.9+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python main.py --mode train \
    --train_list path/to/train.txt \
    --dev_list path/to/dev.txt \
    --model {aasist,beats,w2v2} \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 0.0001 \
    --output_dir ./exp
```

### Evaluation
```bash
python main.py --mode eval \
    --eval_list path/to/eval.txt \
    --model {aasist,beats,w2v2} \
    --checkpoint path/to/model.pth \
    --output_dir ./exp
```

### Inference
```bash 
python main.py --mode infer \
    --audio path/to/audio.wav \
    --model {aasist,beats,w2v2} \
    --checkpoint path/to/model.pth
```

## Models

This repository contains model implementations for the EnvSDD (Environmental Sound Deepfake Detection) competition. The models focus on detecting audio deepfakes with emphasis on generalization to unseen generators.

Available model variants:
- aasist
- beats
- w2v2

Please refer to the competition guidelines for model implementation details.

## Tips for Better Generalization
1. Use data augmentation (noise, reverb, compression)
2. Train with diverse fake generators
3. Ensemble different model variants
4. Use score calibration for unseen conditions

## Citation
This repository is an implementation for the EnvSDD competition. If you use this code for your research, please cite:
```bibtex
@article{envsdd,
  title={{EnvSDD}: Benchmarking Environmental Sound Deepfake Detection},
  author={Yin, Han and Xiao, Yang and Das, Rohan Kumar and Bai, Jisheng and Liu, Haohe and Wang, Wenwu and Plumbley, Mark D},
  booktitle={Interspeech},
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- AASIST paper and implementation
- BEATs and Wav2Vec2 pre-trained models
- ASVspoof community and datasets