@echo off
cd /d "%~dp0\..\.."
python train\lora_jp\run_pipeline.py %*
