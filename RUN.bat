@echo off
cd /d %~dp0
streamlit run PalletSiteWithlaser.py --server.fileWatcherType=none
pause