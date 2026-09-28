# Prediction_Of_Dynamic_Cloud_Resources_Provisioning_for_Workflow

## Run the UI

This project predicts the next five-minute Azure workload from the previous five readings of minimum, maximum, and average CPU usage.

From the project folder:

```powershell
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training command saves the model and scaler in `artifacts/`. The app accepts edited readings or a CSV upload containing `min cpu`, `max cpu`, and `avg cpu` columns, using the latest five rows as input.
Many companies are utilizing the cloud for their day to day activities. Many big cloud service providers like AWS, Microsoft Azure have been success-fully serving its increasing customer base. A brief understanding of the char-acteristics of production virtual machine (VM) workloads of large cloud pro-viders can inform the providers resource management systems, e.g. VM scheduler, power manager, server health manager. In our project we will be analysing Microsoft Azure’s VM CPU utilization dataset released in October 2017. We predict the VM workload from the CPU usage pattern like mini-mum, maximum and average from the Azure dataset. Different techniques among Deep learning are used for the prediction by considering the history of the workload. By considering real VM traces, we can show that the predic-tion-informed schedules increase utilization and stop physical resource ex-haustion. We can arrive at a conclusion that cloud service providers can use their workloads’ characteristics and machine learning techniques to enhance resource management greatly.
