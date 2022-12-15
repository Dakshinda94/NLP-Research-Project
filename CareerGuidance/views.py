from django.shortcuts import render
from django.http.response import StreamingHttpResponse

from django.contrib import messages



from CareerGuidance import models as Umodels
from django.http import HttpResponseRedirect
import json
from django.http import HttpResponse
from django.template import loader
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3
import numpy as np


# Create your views here.
import pandas as pd



def Emp_registration(request):
    if request.method == 'POST':
        Name = request.POST.get('Name', '')
        Address = request.POST.get('Address', '')
        Contact = request.POST.get('Contact', '')
        Email = request.POST.get('Email', '')
        Password = request.POST.get('Password', '')
        Umodels.Employee.objects.create(Name=Name, Address=Address, Contact=Contact, Email=Email, Password=Password)

    #    messages.success(request, 'Successfully Registered!')
    #    return HttpResponseRedirect(request.path_info)
        return render(request, 'CareerGuidance/Emp_login.html')
    return render(request, 'CareerGuidance/Emp_registration.html')



from django.contrib.auth.views import LoginView

def Emp_login(request):
    if request.method == 'POST':
        Email = request.POST.get('Email', '')
        Password = request.POST.get('Password', '')
        User_found = Umodels.Employee.objects.all().filter(Email=Email, Password=Password).count()
        if User_found > 0:
            request.session['Email'] = Email
            return render(request, 'CareerGuidance/Emp_home0.html', {'Email': request.session['Email']})
        else:
            #	print("Failed")
            messages.success(request, 'Invalid ! Please Check Your UserName and Password')
            return HttpResponseRedirect(request.path_info)
        messages.success(request, 'Successfully Registered!')
        return HttpResponseRedirect(request.path_info)
    return render(request, 'CareerGuidance/Emp_login.html')

def Stu_home0(request):
    return render(request, 'CareerGuidance/Stu_home0.html')

def Stu_home(request):
    courseDataset = pd.read_csv('Course.csv')
    finalpt = courseDataset.head(10)
    json_records = finalpt.reset_index().to_json(orient='records')
    courseDataset = json.loads(json_records)
    print(courseDataset)
    context = {'courseDataset': courseDataset}

    template = loader.get_template('CareerGuidance/Stu_home.html')
    return HttpResponse(template.render(context, request))


def Emp_home0(request):
    return render(request, 'CareerGuidance/Emp_home0.html')

def Emp_home(request):

    JobDataset = pd.read_csv('JobDataset.csv')
    finalpt = JobDataset.head(10)
    json_records = finalpt.reset_index().to_json(orient='records')
    loadjob = json.loads(json_records)
    print(loadjob)
    context = {'loadjob': loadjob}

    template = loader.get_template('CareerGuidance/Emp_home.html')
    return HttpResponse(template.render(context, request))





def Job_profile(request):
    if request.method == 'POST':
        Slected_Industry = request.POST.get('Slected_Industry', '')
        skills = request.POST.get('skills', '')
        Experience = request.POST.get('Experience', '')

        Umodels.jobprofile.objects.create(jobIndustry=Slected_Industry, skill=skills, experience=Experience, Email=request.session['Email'])

        messages.success(request, 'Job Profile Successfully Added!')
        return HttpResponseRedirect(request.path_info)
    return render(request, 'CareerGuidance/Job_profile.html', {'Email':request.session['Email']})



def job_search(request):
    if request.method == 'POST':
        Slected_Industry = request.POST.get('Slected_Industry', '')
        dataset = pd.read_csv('JobDataset.csv')
        dataset = dataset[(dataset["Industry"] == Slected_Industry)]
        finalpt = dataset.head(10)
        json_records = finalpt.reset_index().to_json(orient='records')
        arr = []
        arr = json.loads(json_records)
        context = {'d': arr}
        return render(request, 'CareerGuidance/job_search.html', context)
    return render(request, 'CareerGuidance/job_search.html' )


def Job_Recommendation(request):
    from rake_nltk import Rake
    import nltk
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer
    import warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv('JobDataset.csv')  # Read dataset

    # keep only these  useful columns,
    df = df[['Position', 'Job_Description']]

    # Step 2: data pre-processing
    # to remove punctuations from Job_Description
    df['Job_Description'] = df['Job_Description'].str.replace('[^\w\s]', '')

    # # alternative way to remove punctuations, same result
    import string
    df['Job_Description'] = df['Job_Description'].str.replace('[{}]'.format(string.punctuation), '')

    # to extract key words from Job_Description to a list
    df['Key_words'] = ''  # initializing a new column
    r = Rake()  # use Rake to discard stop words (based on english stopwords from NLTK)

    for index, row in df.iterrows():
        r.extract_keywords_from_text(
            row['Job_Description'])  # to extract key words from Job_Description, default in lower case
        key_words_dict_scores = r.get_word_degrees()  # to get dictionary with key words and their scores
        row['Key_words'] = list(key_words_dict_scores.keys())  # to assign list of key words to new column

    # to see last item in Job_Description
    df['Job_Description'][11]

    # to see last dictionary extracted from Job_Description
    key_words_dict_scores
    # to see last item in Key_words
    df['Key_words'][11]

    # Step 3: create word representation by combining column attributes to Bag_of_words

    # to combine 4 lists (4 columns) of key words into 1 sentence under Bag_of_words column
    df['Bag_of_words'] = ''
    columns = ['Key_words']

    for index, row in df.iterrows():
        words = ''
        for col in columns:
            words += ' '.join(row[col]) + ' '
        row['Bag_of_words'] = words

    # strip white spaces infront and behind, replace multiple whitespaces (if any)
    df['Bag_of_words'] = df['Bag_of_words'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')

    df = df[['Job_Description', 'Bag_of_words']]

    # an example to see what is in the Bag_of_words
    df['Bag_of_words'][0]

    # Step 4: create vector representation for Bag_of_words and the similarity matrix
    # to generate the count matrix
    count = CountVectorizer()
    count_matrix = count.fit_transform(df['Bag_of_words'])
    count_matrix

    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # print(cosine_sim)

    # Step 5: run and test the recommender model

    # to create a Series for job Positions which can be used as indices (each index is mapped to a job Position)
    indices = pd.Series(df['Job_Description'])

    # print(indices)
    # indices[:5]

    # most simliar input

    Recommend_jobs = []

    from fuzzywuzzy import process
    CollectedDataSet = pd.read_csv('JobDataset.csv')
    Job_Description = CollectedDataSet['Job_Description'].tolist()

    # User Job
    getmail = request.session['Email']
    emp_ski_exp = Umodels.jobprofile.objects.filter(Email = getmail).values_list("skill")
    print('TESTTTT')
    emp_ski_exp = list(emp_ski_exp)[-1][0]
    print(emp_ski_exp)

    str2Match = emp_ski_exp
    highest = process.extractOne(str2Match, Job_Description)[0]
    print(highest)
    idx = CollectedDataSet[CollectedDataSet['Job_Description'] == highest].values[0][0]
    print(idx)

    #  idx = indices[indices == Position].index[0]  # to get the index of the job Position matching the input job
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)  # similarity scores in descending order
    top_10_indices = list(score_series.iloc[1:10].index)  # to get the indices of top 10 most similar jobs
    print(top_10_indices)

    M1 = (top_10_indices[0])
    M2 = (top_10_indices[1])
    M3 = (top_10_indices[2])

    dataset = pd.read_csv('JobDataset.csv')  # Read dataset
    M1 = dataset[(dataset["index"] == M1)]
    print(M1[['Position']])
    M1 = M1[['index']].values[0][0]


    print(M1)

    dataset = pd.read_csv('JobDataset.csv')  # Read dataset
    M2 = dataset[(dataset["index"] == M2)]
    print(M2[['Position']])
    M2 = M2[['index']].values[0][0]

    print(M2)

    dataset = pd.read_csv('JobDataset.csv')  # Read dataset
    M3 = dataset[(dataset["index"] == M3)]
    print(M3[['Position']])
    M3 = M3[['index']].values[0][0]


    dataset = pd.read_csv('JobDataset.csv')

    dataset = dataset[dataset["index"].isin([M1, M2 , M3])]

    finalpt = dataset.head(10)
    json_records = finalpt.reset_index().to_json(orient='records')
    arr = json.loads(json_records)
    context = {'d': arr}
    return render(request, 'CareerGuidance/Job_Recommendation.html', context)


# ------------------------------ Stu


def Stu_registration(request):
    if request.method == 'POST':
        Name = request.POST.get('Name', '')
        Address = request.POST.get('Address', '')
        Slected_Industry = request.POST.get('Slected_Industry', '')
        Email = request.POST.get('Email', '')
        Password = request.POST.get('Password', '')
        Umodels.Student.objects.create(StudentName=Name, Address=Address, Interest=Slected_Industry, Email=Email, Password=Password)
     #   messages.success(request, 'Successfully Registered!')
    #    return HttpResponseRedirect(request.path_info)
        return render(request, 'CareerGuidance/Stu_login.html')
    return render(request, 'CareerGuidance/Stu_registration.html')



from django.contrib.auth.views import LoginView

def Stu_login(request):
    if request.method == 'POST':
        Email = request.POST.get('Email', '')
        Password = request.POST.get('Password', '')
        User_found = Umodels.Student.objects.all().filter(Email=Email, Password=Password).count()
        if User_found > 0:
            request.session['Email'] = Email
            return render(request, 'CareerGuidance/Stu_home0.html', {'Email': request.session['Email']})
        else:
            #	print("Failed")
            messages.success(request, 'Invalid ! Please Check Your UserName and Password')
            return HttpResponseRedirect(request.path_info)
        messages.success(request, 'Successfully Registered!')
        return HttpResponseRedirect(request.path_info)
    return render(request, 'CareerGuidance/Stu_login.html')





def Stu_profile(request):
    if request.method == 'POST':
        skills = request.POST.get('skills', '')
        Experience = request.POST.get('Experience', '')

        Umodels.Student_jobprofile.objects.create( skill=skills, experience=Experience, Email=request.session['Email'])

        messages.success(request, 'Job Profile Successfully Added!')
        return HttpResponseRedirect(request.path_info)
    return render(request, 'CareerGuidance/Stu_profile.html', {'Email':request.session['Email']})




def Course_Recommendation(request):
    from rake_nltk import Rake
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer
    import string

    # Course
    getmail = request.session['Email']

    print("lllllll")

    user_CourseName = Umodels.Student.objects.values_list('Interest').filter(Email=getmail)
#    user_CourseName = (user_CourseName[0][0])
 #   print(user_CourseName)


    #  print(user_CourseName)


    CourseDS = pd.read_csv('Course.csv')
  #  CourseDS = CourseDS[CourseDS['Type'] == user_CourseName]

    # Slelect Coumns
    CourseDS = CourseDS[['CourseName', 'Course_content']]

    # clearning Dataset
    CourseDS['Course_content'] = CourseDS['Course_content'].str.replace('[^\w\s]', '')
    CourseDS['Course_content'] = CourseDS['Course_content'].str.replace('[{}]'.format(string.punctuation), '')
    # to extract key words from Job_Description to a list
    CourseDS['Key_words'] = ''  # initializing a new column
    r = Rake()  # use Rake to discard stop words (based on english stopwords from NLTK)

    for index, row in CourseDS.iterrows():
        r.extract_keywords_from_text(
            row['Course_content'])  # to extract key words from Job_Description, default in lower case
        key_words_dict_scores = r.get_word_degrees()  # to get dictionary with key words and their scores
        row['Key_words'] = list(key_words_dict_scores.keys())  # to assign list of key words to new column

    # to see last item in Job_Description
    CourseDS['Course_content'][1]

    # to see last dictionary extracted from Job_Description
    key_words_dict_scores
    # to see last item in Key_words
    CourseDS['Course_content'][1]

    # Step 3: create word representation by combining column attributes to Bag_of_words

    # to combine 4 lists (4 columns) of key words into 1 sentence under Bag_of_words column
    CourseDS['Bag_of_words'] = ''
    columns = ['Key_words']

    for index, row in CourseDS.iterrows():
        words = ''
        for col in columns:
            words += ' '.join(row[col]) + ' '
        row['Bag_of_words'] = words

    # strip white spaces infront and behind, replace multiple whitespaces (if any)
    CourseDS['Bag_of_words'] = CourseDS['Bag_of_words'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')

    CourseDS = CourseDS[['Course_content', 'Bag_of_words']]

    # an example to see what is in the Bag_of_words
    CourseDS['Bag_of_words'][0]

    # Step 4: create vector representation for Bag_of_words and the similarity matrix
    # to generate the count matrix
    count = CountVectorizer()
    count_matrix = count.fit_transform(CourseDS['Bag_of_words'])
    count_matrix

    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # print(cosine_sim)

    # Step 5: run and test the recommender model

    # to create a Series for job Positions which can be used as indices (each index is mapped to a job Position)
    indices = pd.Series(CourseDS['Course_content'])

    # print(indices)
    # indices[:5]

    # most simliar input

    Recommend_jobs = []

    from fuzzywuzzy import process
    CollectedDataSet = pd.read_csv('Course.csv')
    Course_content = CollectedDataSet[CollectedDataSet['Type'] == CourseDS]
    Course_content = CollectedDataSet['Course_content'].tolist()

    # User Job

    getmail = request.session['Email']
    print(getmail)

    emp_ski_exp = Umodels.Student_jobprofile.objects.filter(Email = getmail).values_list("skill")
    print('TESTTTT')
    emp_ski_exp = list(emp_ski_exp)[-1][0]

    print(emp_ski_exp)

    str2Match = emp_ski_exp
    highest = process.extractOne(str2Match, Course_content)[0]
    print(highest)
    idx = CollectedDataSet[CollectedDataSet['Course_content'] == highest].values[0][0]
    print(idx)

    #  idx = indices[indices == Position].index[0]  # to get the index of the job Position matching the input job
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)  # similarity scores in descending order
    top_10_indices = list(score_series.iloc[1:10].index)  # to get the indices of top 10 most similar jobs
    print(top_10_indices)

    M1 = (top_10_indices[0])
    M2 = (top_10_indices[1])
    M3 = (top_10_indices[2])

    dataset = pd.read_csv('Course.csv')  # Read dataset
#    M1 = dataset[(dataset["index"] == M1)]
    print(1111111111111)
    print(M1)


    dataset = pd.read_csv('Course.csv')  # Read dataset
  #  dataset = dataset[dataset['Type'] == CourseDS]
    M1 = dataset[(dataset["index"] == M1)]
    M1 = M1[['index']].values[0][0]
    print(M1)

    dataset = pd.read_csv('Course.csv')  # Read dataset
   #@ dataset = dataset[dataset['Type'] == CourseDS]
    M2 = dataset[(dataset["index"] == M2)]
    M2 = M2[['index']].values[0][0]
    print(M2)



    dataset = pd.read_csv('Course.csv')
 #   dataset = dataset[dataset['Type'] == CourseDS]
    dataset = dataset[dataset["index"].isin([M1, M2, M3 ])]


    finalpt = dataset.head(10)
    json_records = finalpt.reset_index().to_json(orient='records')
    arr = json.loads(json_records)
    context = {'d': arr}
    return render(request, 'CareerGuidance/Course_Recommendation.html', context)
