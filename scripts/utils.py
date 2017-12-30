import json
import time

""" construct a string representation from a regular expression """
def reg2str(reg):
        return reg.replace(".*", "all").replace("*", "").replace(".", "") if reg else "all"

""" log messages in a nice format """
def log(*messages):
        print(time.strftime("[%Y-%m-%d %H:%M:%S]"), ' '.join(messages))

""" save the given parameters as json """
def save_json(path, parameters, indent=4):
        with open(path, "w") as output:
                output.write(json.dumps(parameters, indent=indent))

""" retrieve the next token in between "begin_delim" and "end_delim" """
def get_token(line, begin_delim, end_delim, start = 0):
        begin = line.find(begin_delim, start) + len(begin_delim)
        end = line.find(end_delim, begin)
        return line[begin : end], end

""" """
def get_seconds(delta):
        delta = delta.replace("ms", "/1000")
        delta = delta.replace("s:", "+")
        delta = delta.replace("m:", "*60*60+")
        delta = delta.replace("h:", "*60*60*60+")
        delta = delta.replace("d:", "*24*60*60*60+")
        while (len(delta) > 0) and (delta[0] == '0'):
                delta = delta[1:]
        delta = delta.replace("+00*", "1*")
        delta = delta.replace("+00", "")
        delta = delta.replace("+0", "+")
        return str(eval(delta))

""" parse a log file and extract the optimum values """
def get_log(path):
        with open(path, "r") as file:
                for line in file:
                        if line.find("speed=") < 0:
                                continue
                        # search for the test loss value
                        value, index = get_token(line, "test=", "|", 0)
                        # search for the test error
                        error, index = get_token(line, "|", ",", index)
                        # search for the optimum number of epochs
                        epoch, index = get_token(line, "epoch=", ",", index)
                        # search for the convergence speed
                        speed, index = get_token(line, "speed=", "/s", index)
                        # duration
                        delta, index = get_token(line, "time=", ".", index)
                        return value, error, epoch, speed, get_seconds(delta)
        print("invalid log file <", path, ">")
