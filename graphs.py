import matplotlib.pyplot as plt
import pandas
import base64
import io
import matplotlib
matplotlib.use("Agg")

def return_graph(oldreturn: list, newreturn: list):

    plt.figure(figsize=(12, 6))
    plt.plot(oldreturn, label = 'Current Return')
    plt.plot(newreturn, label = 'Optimized Return')
    plt.title('Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True)
    plt.legend(loc="lower right")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    buffer_value = buf.read()
    image_base64 = base64.b64encode(buffer_value).decode('utf-8').replace("\n", "")

    return image_base64