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

namespace GameOfLife.View.Controls {
    /// <summary>
    /// Interaction logic for CellUniverseView.xaml
    /// </summary>
    public partial class CellUniversePanel : UserControl {
        private CellUniverse _cellUniverse;

        private bool[,] _cells;

        public CellUniversePanel() {

            InitializeComponent();

            _cellUniverse = new CellUniverse();

            System.Drawing.Bitmap source = new System.Drawing.Bitmap("convay.png");

            var myCells = BitmapOps.ToBoolArray(BitmapOps.DownScale(source, 2), 1);

            AddRectangles(myCells);
            _cells = myCells;

            StartGame();
        }

        private void StartGame() {
            DispatcherTimer timer = new DispatcherTimer();
            timer.Interval = new TimeSpan(640000);
            timer.Tick += (s, e) => { DoIt(); };
            timer.Start();
        }

        private void DoIt() {
            var cells = _cellUniverse.NewGeneration(_cells);
            _cells = cells;
            AddRectangles(cells);
        }

        private Thickness GetMargin(int row, int column) {
            //cellUniverseContainer.Width;
            //cellUniverseContainer.Height;
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