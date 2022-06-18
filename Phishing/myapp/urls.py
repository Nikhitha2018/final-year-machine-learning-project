# creating URLs for accessing the views written in the views.py file in Django project

from django.urls import path
from . import views
urlpatterns = [ 
    path('',views.index,name='index'),
    path('ind',views.ind,name='ind'),
    path('api',views.api,name='api'),
    path('search',views.search,name="search"),
    path('result',views.result,name='result'),
    path('detect',views.detect,name='detect'),
    #path('aendetect',views.aendetect,name='aendetect'),
    path('xgboostresult',views.xgboostresult,name='xgboostresult'),
    path('about',views.about,name='about'),
    path('home',views.home,name='home'),
    path('geturlhistory',views.geturlhistory,name="geturlhistory"),
    path('page',views.page,name='page'),
    path('addcourse',views.addcourse,name="addcourse"),
    path('showblack',views.showblack,name="showblack"),
    path('updatee_view/<int:id>',views.updatee_view,name='updatee_view' ),
    path('de_view/<int:id>',views.de_view,name='de_view'),   
    #path('aenresult/',views.aenresult,name='aenresult'),
]

