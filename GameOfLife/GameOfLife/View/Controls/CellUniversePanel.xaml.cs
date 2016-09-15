using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
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

    using Model;

    public partial class CellUniversePanel : UserControl {

        private WriteableBitmap _writeableBitmap;
        private Image _image;

        private int _backgroundColor;
        private int _foregroundColor;

        private int _width  = 160;
        private int _height = 102;

        private int _surfaceWidth  = 0;
        private int _surfaceHeight = 0;

        private CellUniverse _cellUniverse;

        private bool[,] _cells;


        public CellUniversePanel() {
            InitializeComponent();
            Initialize();
            StartGame();
        }

        private void Initialize() {

            _image = new Image();

            RenderOptions.SetBitmapScalingMode(_image, BitmapScalingMode.NearestNeighbor);
            RenderOptions.SetEdgeMode(_image, EdgeMode.Aliased);

            cellUniverseSurface.Children.Add(_image);

            _backgroundColor = GetIntColor(0, 0, 0);
            _foregroundColor = GetIntColor(0, 122, 204);

            _cells = new bool[_height, _width];

            Random random = new Random();

            for (int y = 0; y < _cells.GetLength(0); y++) {
                for (int x = 0; x < _cells.GetLength(1); x++) {
                    _cells[y, x] = random.Next(2) == 1;
                }
            }

            _cellUniverse = new CellUniverse();
        }

        private int GetIntColor(byte red, byte green, byte blue) {
            int result = red << 16;
            result |= green << 8;
            result |= blue << 0;
            return result;
        }

        private void StartGame() {
            DispatcherTimer timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromMilliseconds(1);
            timer.Tick += (s, e) => { DoIt(); };
            timer.Start();
        }

        private void DoIt() {

            LazyInitialization();

            var cells = _cellUniverse.NewGeneration(_cells);
            var difference = GetDifference(_cells, cells);
            Redraw(cells, difference);
            //AddRectangles(cells);
            _cells = cells;
        }

        private void LazyInitialization() {

            if (_surfaceWidth != (int)ActualWidth || _surfaceHeight != (int)ActualHeight) {
                _surfaceWidth = (int)ActualWidth;
                _surfaceHeight = (int)ActualHeight;
            }
            if (_writeableBitmap == null || _writeableBitmap.Width != _surfaceWidth || _writeableBitmap.Height != _surfaceHeight) {
                _writeableBitmap = new WriteableBitmap(_surfaceWidth, _surfaceHeight, 96, 96, PixelFormats.Bgr32, null);
                _image.Source = _writeableBitmap;
                _image.Stretch = Stretch.None;
            }
        }

        private bool[,] GetDifference(bool[,] oldArr, bool[,] newArr) {

            if (oldArr.GetLength(0) != newArr.GetLongLength(0) || oldArr.GetLength(1) != newArr.GetLongLength(1)) {
                throw new InvalidOperationException();
            }

            bool[,] result = new bool [oldArr.GetLength(0), oldArr.GetLength(1)];

            for (int y = 0; y < oldArr.GetLength(0); ++y) {
                for (int x = 0; x < oldArr.GetLength(1); ++x) {
                    if (oldArr[y, x] != newArr[y, x]) {
                        result[y, x] = true;
                    }
                }
            }
            return result;
        }

        private void Redraw(bool[,] cells, bool[,] difference) {

            int cellSize = GetCellSize();
            int offsetX = ((int)ActualWidth - (cells.GetLength(1) * cellSize + cells.GetLength(1))) / 2;
            int offsetY = ((int)ActualHeight - (cells.GetLength(0) * cellSize + cells.GetLength(0))) / 2;

            Random random = new Random();

            for (int y = 0; y < cells.GetLength(0); ++y) {
                for (int x = 0; x < cells.GetLength(1); ++x) {

                    if (cells[y, x] && difference[y, x]) {
                        DrawRect(
                            _writeableBitmap,
                            offsetX + x * cellSize + x, offsetY + y * cellSize + y, cellSize, cellSize,
                            _foregroundColor
                            //GetIntColor((byte)random.Next(256), (byte)random.Next(256), (byte)random.Next(256))
                            //GetIntColor(50, 160, 250)
                            //GetIntColor((byte)(255 - x), (byte)(255 - x), (byte)(255 - y))
                            );
                    }
                    if (!cells[y, x] && difference[y, x]) {
                        DrawRect(
                            _writeableBitmap,
                            offsetX + x * cellSize + x, offsetY + y * cellSize + y, cellSize, cellSize,
                            _backgroundColor
                            );
                    }
                }
            }
        }

        private int GetCellSize() {
            int horizontalRatio = (_surfaceWidth - _width) / _width;
            int verticalRatio = (_surfaceHeight - _height) / _height;
            return verticalRatio < horizontalRatio ? verticalRatio : horizontalRatio;
        }

        private void DrawRect(WriteableBitmap bitmap, int posX, int posY, int width, int height, int color) {

            if (posX + width > _surfaceWidth || posY + height > _surfaceHeight) {
                return;
            }

            bitmap.Lock();
            unsafe
            {
                for (int y = posY; y < posY + height; ++y) {
                    for (int x = posX; x < posX + width; ++x) {

                        int pBackBuffer = (int)_writeableBitmap.BackBuffer;

                        pBackBuffer += y * bitmap.BackBufferStride;
                        pBackBuffer += x * 4;

                        *((int*)pBackBuffer) = color;
                    }
                }
                bitmap.AddDirtyRect(new Int32Rect(posX, posY, width, height));
            }
            bitmap.Unlock();
        }

        //private void AddRectangles(bool[,] cells) {

        //    cellUniverseSurface.Children.Clear();

        //    for (int x = 0; x < cells.GetLength(0); x++) {
        //        for (int y = 0; y < cells.GetLength(1); y++) {
        //            if (!cells[x, y])
        //                continue;

        //            Rectangle rectangle = new Rectangle();
        //            rectangle.Height = 4;
        //            rectangle.Width = 4;
        //            rectangle.Fill = new SolidColorBrush(Color.FromRgb(0, 127, 192));
        //            rectangle.VerticalAlignment = VerticalAlignment.Top;
        //            rectangle.HorizontalAlignment = HorizontalAlignment.Left;

        //            rectangle.Margin = GetMargin(x, y);

        //            cellUniverseSurface.Children.Add(rectangle);
        //        }
        //    }
        //}

        //private Thickness GetMargin(int row, int column) {
        //    return new Thickness(column * 5, row * 5, 0, 0);
        //}
    }
}