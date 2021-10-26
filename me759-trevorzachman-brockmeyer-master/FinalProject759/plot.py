import matplotlib.pyplot as plt

def save_plot(*args, **kwargs):
    x_start = kwargs['x_start']
    x_end = kwargs['x_end']

    title = kwargs['title']
    x_label = kwargs['x_label']
    y_label = kwargs['y_label']

    filename = kwargs['filename']

    plt.xticks(range(x_start, x_end+1))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend()

    plt.savefig(filename)  


def plot(*args, **kwargs):
    data_files = kwargs['data_files']

    for i, fi in enumerate(data_files):
        x_start = kwargs['x_start']
        x_end = kwargs['x_end']
        x = range(x_start, x_end+1)

        color = kwargs['colors'][i]
        label = kwargs['labels'][i]

        file_lines = []
        data = []
        with open(fi) as f:
            file_lines = f.read().splitlines()

        for i in range(len(file_lines)):
            data.append(file_lines[i])

        data = [float(d) for d in data]

        plt.plot(x, data, linestyle='--', color=color, marker='o', label=label)
    

def main():
    plots = {
        'data_files': ['omp_for.out', 'omp_task.out', 'serial_miner.out'],
        'x_start': 5,
        'x_end': 25,
        'colors': ['red', 'blue', 'green'],
        'labels': ['omp for loop sharing, threads=20', 'omp task sharing, threads=20', 'serial miner']
    }

    save = {
        'x_start': 5,
        'x_end': 25,
        'title': 'Bitcoin Mining Time Analysis',
        'x_label': 'n, where #iterations = 2^n ',
        'y_label': 'time',
        'filename': 'serial_miner_time_analysis.pdf',
    }

    plot(**plots)
    save_plot(**save)
    

main()
