"""Training entry point. Owner: Abdallah Afifi

Usage: python scripts/train.py --config configs/centernet_voc5.yaml --out runs/exp1

Writes: runs/<exp>/best.pt, last.pt, history.csv (epoch, train_loss parts, val_loss, val_mAP),
        loss_curve.png. Must resume from last.pt (Kaggle/Colab sessions time out).
"""
