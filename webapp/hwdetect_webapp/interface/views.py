from django.shortcuts import render

def interface(request):
    return render(request, 'interface/interface.html', locals())