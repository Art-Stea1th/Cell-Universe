using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace GameOfLife {

    public partial class MainWindow : Window {

        private CellUniverse _cellUniverse;
        private bool[,] _cells;

        public MainWindow() {

            InitializeComponent();

            _cellUniverse = new CellUniverse();

            var myCells = new bool[100, 100];

            //for (int i = 0; i < myCells.GetLength(0); i++) {
            //    for (int j = 0; j < myCells.GetLength(1); j++) {
            //        myCells[i, j] = true;
            //    }
            //}

            myCells[5, 5] = true;
            myCells[6, 4] = true;
            myCells[6, 5] = true;
            myCells[6, 6] = true;
            myCells[7, 6] = true;

            myCells[9, 9] = true;
            myCells[10, 8] = true;
            myCells[10, 9] = true;
            myCells[10, 10] = true;
            myCells[11, 10] = true;

            myCells[25, 25] = true;
            myCells[26, 24] = true;
            myCells[26, 25] = true;
            myCells[26, 26] = true;
            myCells[27, 26] = true;

            myCells[35, 35] = true;
            myCells[36, 34] = true;
            myCells[36, 35] = true;
            myCells[36, 36] = true;
            myCells[37, 36] = true;

            myCells[39, 39] = true;
            myCells[40, 38] = true;
            myCells[40, 39] = true;
            myCells[40, 40] = true;
            myCells[41, 40] = true;

            myCells[55, 55] = true;
            myCells[56, 54] = true;
            myCells[56, 55] = true;
            myCells[56, 56] = true;
            myCells[57, 56] = true;

            AddRectangles(myCells);
            _cells = myCells;

            StartGame();
        }

        private void StartGame() {
            DispatcherTimer timer = new DispatcherTimer();
            timer.Interval = new TimeSpan(240000);
            timer.Tick += (s, e) => { DoIt(); };
            timer.Start();
        }

        private void DoIt() {
            var cells = _cellUniverse.NewGeneration(_cells);
            _cells = cells;
            AddRectangles(cells);
        }

        private Thickness GetMargin(int row, int column) {
            return new Thickness(column * 5, row * 5, 0, 0);
        }

        private void AddRectangles(bool[,] cells) {

            cellUniverseContainer.Children.Clear();

            for (int x = 0; x < cells.GetLength(0); x++) {
                for (int y = 0; y < cells.GetLength(1); y++) {
                    if (!cells[x, y])
                        continue;

                    Rectangle rectangle = new Rectangle();
                    rectangle.Height = 4;
                    rectangle.Width = 4;
                    rectangle.Fill = new SolidColorBrush(Color.FromRgb(0, 127, 192));
                    rectangle.VerticalAlignment = VerticalAlignment.Top;
                    rectangle.HorizontalAlignment = HorizontalAlignment.Left;

                    rectangle.Margin = GetMargin(x, y);

                    cellUniverseContainer.Children.Add(rectangle);
                }
            }
        }

        
    }
}