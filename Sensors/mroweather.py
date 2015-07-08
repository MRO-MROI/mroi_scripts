for line in open('WX20120308.log', 'r'):
    stripspaces = " ".join(line.split())
    splitline = stripspaces.split(' ')
    timetemp =  splitline[3], splitline[4], splitline[5], (float(splitline[10])+float(splitline[23])+float(splitline[24]))/3.0, (float(splitline[26])+float(splitline[27])+float(splitline[28]))/3.0 
    
    
    