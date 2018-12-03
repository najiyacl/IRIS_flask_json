# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:37:59 2018

@author: najiyacl
"""

import json
import requests
import pandas as pd
"""Setting the headers to send and accept json responses
"""
header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}


sepal_length=3
sepal_width=3
petal_length=1
petal_width=2

requestData = pd.DataFrame({'sepal_length':[sepal_length],'sepal_width':[sepal_width],
                            'petal_length':[petal_length],'petal_width':[petal_width]})

data = requestData.to_json(orient='records')

resp= requests.post("http://localhost:5000/predict", \
                    data = json.dumps(data),\
                    headers= header)

print(resp.json())

