import webbrowser

def reportgen(filename="test", network=[784,16,16,10], no_iter=100, learn_rate=0.05,
              train_acc=1, test_acc=1, cost_f=1, lamda=0, note="", show_after=True):
    f = open(filename+".html", "w")
    f.writelines("<HTML>\n<HEAD>\n<title>" + "Report: " + str(filename) + "</title>")
    f.writelines("\n<style>h1 {text-align: center;}</style>\n</HEAD>\n")
    f.writelines("\n<BODY>")
    f.writelines("<H1> Report Name: " + filename + "</H1>\n")
    f.writelines("<H3>Neural Network Layer Structure is: "+str(network)+"</H3>\n")
    f.writelines("<H3>Number of Iterations done are: " + str(no_iter) + "</H3>\n")
    f.writelines("<H3>Learning Rate used is: " + str(learn_rate) + "</H3>\n")
    f.writelines("<H3>Train set accuracy is: " + str(train_acc*100) + "%</H3>\n")
    f.writelines("<H3>Test set accuracy is: " + str(test_acc*100) + "%</H3>\n")
    f.writelines("<H3>Lambda used is: " + str(lamda) + "</H3>\n")
    f.writelines("<H3>Final Cost is: " + str(cost_f) + "</H3>\n")
    if note != "":
        f.writelines("<H3>Special Notes: "+note+"</H3>\n")
    f.writelines("<H3>Cost Function Graph is: </H3>")
    f.writelines('<center><img src="'+filename+'.png" alt="Cost Function Graph"></center>')
    f.writelines("\n</BODY>\n</HTML>\n")
    f.close()
    webbrowser.open(filename+".html")
    return