from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from agg import *


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mainWidget = WindowClass(parent=self)
        self.setCentralWidget(self.mainWidget)
        bar = self.menuBar()
        options_menu = bar.addMenu('Opcje')
        about_action = QAction('O programie', self)
        close_action = QAction('Zamknij', self)
        options_menu.addAction(close_action)
        options_menu.addAction(about_action)
        close_action.triggered.connect(self.close)
        about_action.triggered.connect(self.about)

    def about(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("This is a message box")
        msg.setInformativeText("This is additional information")
        msg.setWindowTitle("MessageBox demo")
        msg.setDetailedText("The details are as follows:")
        msg.exec_()

class WindowClass(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.button_start = None
        self.figure = None
        self.canvas = None
        self.number_of_object = 30
        self.number_of_iterations = 4
        self.number_of_processes = 1
        self.graph = None
        self.graph_numbers = None
        self.costs = None
        self.times = None
        self.distances = None
        self.cost_avg = None
        self.basic_generation = None
        self.number_of_nodes = 5
        self.max_number_of_vertices = 24
        self.initUI()

    def initUI(self):
        self.setGeometry(1000, 100, 100, 100)
        self.setWindowTitle('Algorytm genetyczny')

        grid_layout = QGridLayout()
        self.setLayout(grid_layout)

        self.label_number_of_vertex = QLabel("Ilość wierzchołków grafu")
        grid_layout.addWidget(self.label_number_of_vertex, 1, 1)
        self.spinBox = QSpinBox()
        self.spinBox.setMaximum(self.max_number_of_vertices)
        grid_layout.addWidget(self.spinBox, 1, 2)

        self.button_start = QPushButton("Start")
        self.button_start.clicked.connect(self.start_function)
        grid_layout.addWidget(self.button_start, 1, 3)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        grid_layout.addWidget(self.canvas, 1, 4)

        self.label_result = QLabel()
        grid_layout.addWidget(self.label_result, 3, 1)

        self.label_result_desc = QLabel("----  Znaleziona ścieżka  ----")
        grid_layout.addWidget(self.label_result_desc, 2, 1)

        self.button_plot = QPushButton("Narysuj statystyki")
        self.button_plot.clicked.connect(self.draw_plots)
        grid_layout.addWidget(self.button_plot, 2, 3)

        self.show()

    def start_function(self):
        random.seed(time.time())
        number_of_nodes = self.spinBox.value()
        if number_of_nodes > 0:
            self.number_of_nodes = number_of_nodes
            self.graph, self.graph_numbers, self.costs, self.times, self.distances = generate_graph(
                self.number_of_nodes)
            self.draw_graph()

            self.basic_generation = generate_population(self.number_of_object, self.number_of_nodes)
            self.cost_avg = []
            self.lack_of_path_avg = []

            for iterator in range(0, self.number_of_iterations):
                # print(iterator)
                self.new_generation = draw_posterity(self.basic_generation)
                self.new_generation = draw_posterity(self.basic_generation)
                self.generation_after_crossover = iteration_of_crossover(self.new_generation)
                self.generation_after_mutation = iteration_of_mutation(self.generation_after_crossover, iterator, self.number_of_nodes,
                                                                  self.number_of_processes)
                self.basic_generation, self.cost, self.lack_of_path = function_of_adaptation(self.generation_after_mutation, self.graph_numbers, self.costs,
                                                                                             self.times, self.distances, self.number_of_object)
                self.cost_avg.append(self.cost)
                self.lack_of_path_avg.append(self.lack_of_path)
            self.label_result.setText(str(self.basic_generation[0].parameters))
            print(print("----  Znaleziona ścieżka  ----"))
            print(self.basic_generation[0].parameters)

    def draw_plots(self):
        if self.basic_generation is not None:
            self.figure.clf()
            ax = self.figure.add_subplot(211)
            ax.plot(self.cost_avg)
            ax.set_title('Średnia wartość funkcji przystosowania')
            plt.subplot(212)
            ax2 = self.figure.add_subplot(212)
            ax2.plot(self.lack_of_path_avg)
            ax2.set_title('Średnia ilość ścieżek nie do przejścia')
            self.canvas.draw_idle()


    def draw_graph(self,
                   node_size=1600, node_color='pink', node_alpha=0.3,
                   node_text_size=12,
                   edge_color='blue', edge_alpha=0.3, edge_tickness=1,
                   edge_text_pos=0.3,
                   text_font='sans-serif'):
        self.figure.clf()
        g = nx.Graph()
        for edge in self.graph:
            g.add_edge(edge[0], edge[1])
        graph_pos = nx.shell_layout(g)
        nx.draw_networkx_nodes(g, graph_pos, node_size=node_size,
                               alpha=node_alpha, node_color=node_color)
        nx.draw_networkx_edges(g, graph_pos, width=edge_tickness,
                               alpha=edge_alpha, edge_color=edge_color)
        nx.draw_networkx_labels(g, graph_pos, font_size=node_text_size,
                                font_family=text_font)
        edge_labels = dict(zip(self.graph, self.costs))
        nx.draw_networkx_edge_labels(g, graph_pos, edge_labels=edge_labels,
                                     label_pos=edge_text_pos)
        self.canvas.draw_idle()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen = MainWindow()
    screen.show()
    sys.exit(app.exec_())
