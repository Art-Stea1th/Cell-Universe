using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Controls;
using System.Windows.Threading;
using System.Windows.Media.Imaging;

namespace CellUniverse.ViewModels {
    
    using Models;
    using Infrastructure;

    public class BinaryViewModel : ViewModelBase {

        private int _width  = 200;
        private int _height = 125;

        private BinaryModel _binaryModel;

        private int _surfaceWidth, _surfaceHeight;

        private int _cellSize;
        private int _offsetX, _offsetY;

        private int _backgroundColor, _foregroundColor;

        private DispatcherTimer _timer;
        private Canvas _cellUniverseSurface;

        internal CellUniverseState CurrentState { get; set; } = CellUniverseState.Stopped;

        internal WriteableBitmap CellUniverseSurfaceContent { get; set; }

        internal void Start() {
            _timer.Start();
            CurrentState = CellUniverseState.Started;
            Update();
        }

        internal void Pause() {
            _timer.Stop();
            CurrentState = CellUniverseState.Paused;
            Update();
        }

        internal void Stop() {
            _timer.Stop();
            Initialize();
            Update();
        }

        internal void SetTimerInterval(TimeSpan interval) {
            _timer.Interval = interval;
        }

        private void Update() {
            InvalidateView();
            if (CurrentState == CellUniverseState.Started) {
                _binaryModel.Next();
            }            
        }

        internal void InvalidateView() {
            ViewRecalculate((int)_cellUniverseSurface.ActualWidth, (int)_cellUniverseSurface.ActualHeight);
            (_cellUniverseSurface.Children[0] as Image).Source = CellUniverseSurfaceContent;
            (_cellUniverseSurface.Children[0] as Image).Stretch = Stretch.None;
            if (CurrentState == CellUniverseState.Paused) {
                _binaryModel.SimulateChangedAll();
            }
        }

        public BinaryViewModel(Canvas cellUniverseSurface) {
            _cellUniverseSurface = cellUniverseSurface;
            Initialize();
        }

        private void Initialize() {

            CellUniverseSurfaceContent = new WriteableBitmap(1, 1, 96, 96, PixelFormats.Bgr32, null);

            _backgroundColor = GetIntColor(0, 0, 0);
            _foregroundColor = GetIntColor(0, 122, 204);

            _binaryModel = new BinaryModel(_width, _height);
            _binaryModel.OnCellChanged += DrawRect;

            CurrentState = CellUniverseState.Stopped;
            _timer = new DispatcherTimer();
            _timer.Tick += (s, e) => { Update(); };
        }

        private int GetIntColor(byte red, byte green, byte blue) {
            int result = red << 16;
            result |= green << 8;
            result |= blue << 0;
            return result;
        }

        private void ViewRecalculate(int actualWidth, int actualHeight) {

            if (_surfaceWidth != actualWidth || _surfaceHeight != actualHeight) {
                _surfaceWidth = actualWidth;
                _surfaceHeight = actualHeight;
            }
            if (CellUniverseSurfaceContent.Width != _surfaceWidth || CellUniverseSurfaceContent.Height != _surfaceHeight) {
                CellUniverseSurfaceContent = new WriteableBitmap(_surfaceWidth, _surfaceHeight, 96, 96, PixelFormats.Bgr32, null);
            }
            if (_cellSize != GetCellSize()) {
                _cellSize = GetCellSize();
                _offsetX = (_surfaceWidth - (_width * _cellSize + _width)) / 2;
                _offsetY = (_surfaceHeight - (_height * _cellSize + _height)) / 2;
            }
        }

        private int GetCellSize() {
            int horizontalRatio = (_surfaceWidth - _width) / _width;
            int verticalRatio = (_surfaceHeight - _height) / _height;
            return verticalRatio < horizontalRatio ? verticalRatio : horizontalRatio;
        }

        private void DrawRect(Tuple<int, int, bool> newState) {
            DrawCell(
                    CellUniverseSurfaceContent,
                    _offsetX + newState.Item1 * _cellSize + newState.Item1,
                    _offsetY + newState.Item2 * _cellSize + newState.Item2,
                    _cellSize, _cellSize,
                    newState.Item3 == true ? _foregroundColor : _backgroundColor);
        }

        private void DrawCell(WriteableBitmap targetBitmap, int startPosX, int startPosY, int width, int height, int color) {

            if (startPosX + width > _surfaceWidth || startPosY + height > _surfaceHeight) {
                return;
            }

            targetBitmap.Lock();
            unsafe
            {
                for (int y = startPosY; y < startPosY + height; ++y) {
                    for (int x = startPosX; x < startPosX + width; ++x) {

                        int pBackBuffer = (int)CellUniverseSurfaceContent.BackBuffer;

                        pBackBuffer += y * targetBitmap.BackBufferStride;
                        pBackBuffer += x * 4;

                        *((int*)pBackBuffer) = color;
                    }
                }
                targetBitmap.AddDirtyRect(new Int32Rect(startPosX, startPosY, width, height));
            }
            targetBitmap.Unlock();
        }
    }
}