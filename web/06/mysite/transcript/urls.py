from django.urls import path, re_path

from . import views


urlpatterns = [
    path('student/list/', views.list_student, name='list_student'),
    re_path('student/update/(?P<sn>\d{6})/$', views.upt_student, name='upt_student'),
]