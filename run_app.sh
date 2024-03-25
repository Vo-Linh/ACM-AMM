export NVIDIA_API_KEY="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"
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=700
streamlit run main.py --server.fileWatcherType none
