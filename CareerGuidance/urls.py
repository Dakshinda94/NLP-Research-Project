from django.urls import path, include
from CareerGuidance import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.Emp_registration, name='Emp_registration'),
    path('Emp_registration', views.Emp_registration, name='Emp_registration'),

    path('Emp_login', views.Emp_login, name='Emp_login'),
   # path('Emp_home', views.Emp_home, name='Emp_home'),

    path('Emp_home/', views.Emp_home),
    path('Emp_home0/', views.Emp_home0),
    path('Stu_home0/', views.Stu_home0),
    path('Stu_home/', views.Stu_home),

    path('Job_profile', views.Job_profile, name='Job_profile'),

    path('job_search', views.job_search, name='job_search'),
    path('Job_Recommendation', views.Job_Recommendation, name='Job_Recommendation'),

    path('Stu_login', views.Stu_login, name='Stu_login'),
    path('Stu_registration', views.Stu_registration, name='Stu_registration'),
    path('Stu_profile', views.Stu_profile, name='Stu_profile'),
    path('Course_Recommendation', views.Course_Recommendation, name='Course_Recommendation'),
    ]
