## Streamlit 

- Usage 
<pre><code>export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024
streamlit run app.py --server.port 30010 --server.fileWatcherType none</code>
</pre>

----    
## Mode
- Output Mode    
    predict와 Ground Truth의 차이를 plot

- GradCAM Mode    
    클래스별 GradCAM plot

- CompareLoss Mode    
    클래스별 binary loss plot