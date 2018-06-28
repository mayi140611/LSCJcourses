# -*- encoding=utf-8 -*-

from django.shortcuts import render
from django.http import Http404, HttpResponse, HttpResponseRedirect

from .models import Student
# Create your views here.


def list_student(request):
    objs = Student.objects.all()
    return render(request, 'list_student.html', {'objs': objs})


def upt_student(request, sn):
    if request.method == 'POST':
        name = request.POST.get('name', '')
        sex = request.POST.get('sex', 0)
        age = request.POST.get('age', 0)
        phone = request.POST.get('phone', '')
        if not name or not sex or not age:
            raise Http404
        try:
            sex = int(sex)
            age = int(age)
        except ValueError:
            raise Http404
        try:
            obj = Student.objects.get(sn=sn)
        except Student.DoesNotExist:
            raise Http404
        obj.name = name
        obj.sex = sex
        obj.age = age
        obj.phone = phone
        obj.save()
        return HttpResponseRedirect('/transcript/student/list/')
    else:
        try:
            obj = Student.objects.get(sn=sn)
        except Student.DoesNotExist:
            raise Http404
        return render(request, 'upt_student.html', {'obj': obj})


