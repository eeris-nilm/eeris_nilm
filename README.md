# eeRIS-NILM: Real-time feedback through NILM

The goal of eeRIS-NILM is to develop and test Non-Intrusive Load Monitoring (NILM) algorithms that help provide electrical energy usage feedback to consumers. The goal is to develop and evaluate algorithms that

 - Are unsupervised, i.e. do not require labeled data for training (but may request feedback from users to assign names to appliances)
 - Operate in near real-time
 
The current implementation provides a relatively simple baseline that assumes two parallel processes. A real-time algorithm that quickly provides "Live" information on appliance usage and energy consumption and an off-line process that is more robust and can be used for summary reports and for analyzing the overall electricity usage of an installation.

For more details, please see our paper at the UPEC 2020 conference:

> Christos Diou and Georgios Andreou, "eeRIS-NILM: An Open Source, Unsupervised Baseline for Real-Time Feedback Through NILM", In Proceedings of the 55th International Universitites Power Engineering Conference (UPEC 2020), Sep. 1-4, 2020

Also, please consider citing this paper if you use this code in your work.

## Try it out

You can check the "Live" algorithm in operation by running an example demo. After cloning the repository, edit the file `tests/demo_animation.py` to fit your setup and location of your datasets. Currently, eeRIS-NILM supports `redd` and `eco` datasets, as well as two internal dataset formats (to be documented). Change to the cloned directory and run the demo via

```bash
$ PYTHONPATH="." python tests/demo_animation.py
```

## Deployment

eeRIS-NILM can be deployed for test in working environments. APIs based on MQTT and on REST are supported. Please refer to the files in the `ini` folder for project configuration.

## Documentation

Check back soon for a sphinx-based API documentation and detailed deployment instructions
