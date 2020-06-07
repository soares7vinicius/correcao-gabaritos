#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(" pwd")


# In[2]:


from pdf2image import convert_from_path, convert_from_bytes

path = "../samples/model3"
images = convert_from_path(path + "/scan.pdf")
for i, image in enumerate(images, 1):
    image.save(path + f"/{i}.png", "PNG")


# In[ ]:
