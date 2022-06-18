from django.contrib import admin
from .models import *

# Models created in Django are registered here 
admin.site.register(UserFeedBack)
admin.site.register(Url)