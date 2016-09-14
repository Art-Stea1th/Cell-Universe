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

            _cellUniverse = new CellUniverse();
        }

        private void StartGame() {
            DispatcherTimer timer = new DispatcherTimer();
            timer.Interval = new TimeSpan(2000);
            timer.Tick += (s, e) => { Redraw(); };
            timer.Start();
        }

        private void Redraw() {
            LazyInitialization();
            DrawRect(_writeableBitmap, 0, 0, _surfaceHeight, _surfaceWidth, GetIntColor(0, 128, 255));
        }

        private int GetIntColor(byte red, byte green, byte blue) {
            int result = red << 16;
            result |= green << 8;
            result |= blue << 0;
            return result;
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

        private void DrawRect(WriteableBitmap bitmap, int posX, int posY, int rows, int columns, int color) {

            Random random = new Random();

            // Compute the pixel's color.
            int color_data = random.Next(256) << 16; // R
            color_data |= random.Next(256) << 8;   // G
            color_data |= random.Next(256) << 0;   // B

            color = color_data;

            bitmap.Lock();
            unsafe
            {
                for (int y = posY; y < posY + rows; ++y) {
                    for (int x = posX; x < posX + columns; ++x) {

                        int pBackBuffer = (int)_writeableBitmap.BackBuffer;

                        pBackBuffer += y * bitmap.BackBufferStride;
                        pBackBuffer += x * 4;

                        *((int*)pBackBuffer) = color;
                    }
                }
                bitmap.AddDirtyRect(new Int32Rect(posX, posY, columns, rows));
            }
            bitmap.Unlock();
        }

        private void DrawPixel() {

            LazyInitialization();

            Random rand = new Random();


            // --------------------

            int column = rand.Next(_surfaceWidth);
            int row = rand.Next(_surfaceHeight);

            // Reserve the back buffer for updates.
            _writeableBitmap.Lock();


            // Get a pointer to the back buffer.
            int pBackBuffer = (int)_writeableBitmap.BackBuffer;

            // Find the address of the pixel to draw.
            pBackBuffer += row * _writeableBitmap.BackBufferStride;
            pBackBuffer += column * 4;

            // Compute the pixel's color.
            int color_data = 0 << 16; // R
            color_data |= 128 << 8;   // G
            color_data |= 255 << 0;   // B

            unsafe
            {
                // Assign the color data to the pixel.
                *((int*)pBackBuffer) = color_data;
            }

            // Specify the area of the bitmap that changed.
            _writeableBitmap.AddDirtyRect(new Int32Rect(column, row, 1, 1));

            // Release the back buffer and make it available for display.
            _writeableBitmap.Unlock();
        }

        private void DoIt() {
            var cells = _cellUniverse.NewGeneration(_cells);
            _cells = cells;
            AddRectangles(cells);
        }

        private void AddRectangles(bool[,] cells) {

            cellUniverseSurface.Children.Clear();

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

                    cellUniverseSurface.Children.Add(rectangle);
                }
            }
        }

        private Thickness GetMargin(int row, int column) {
            return new Thickness(column * 5, row * 5, 0, 0);
        }
    }
}