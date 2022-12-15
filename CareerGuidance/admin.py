from django.contrib import admin

from .models import Employee
admin.site.register(Employee)

from .models import jobprofile
admin.site.register(jobprofile)

from .models import Student
admin.site.register(Student)

from .models import Student_jobprofile
admin.site.register(Student_jobprofile)
