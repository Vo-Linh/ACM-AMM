export HUGGINGFACEHUB_API_TOKEN="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"
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=700
streamlit run main.py --server.fileWatcherType none
