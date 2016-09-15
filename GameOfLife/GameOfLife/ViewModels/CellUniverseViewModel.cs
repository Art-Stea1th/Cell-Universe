using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace GameOfLife.ViewModels {

    using Models;

    public class CellUniverseViewModel : ViewModelBase {

        private int _width  = 160;
        private int _height = 102;

        private CellUniverseModel _cellUniverseModel;

        private int _surfaceWidth;
        private int _surfaceHeight;

        private int _cellSize;
        private int _offsetX;
        private int _offsetY;

        private int _backgroundColor;
        private int _foregroundColor;

        public WriteableBitmap CellUniverseWritableBitmap { get; set; }

        public CellUniverseViewModel() {
            Initialize();
        }

        private void Initialize() {

            CellUniverseWritableBitmap = new WriteableBitmap(1, 1, 96, 96, PixelFormats.Bgr32, null);

            _backgroundColor = GetIntColor(0, 0, 0);
            _foregroundColor = GetIntColor(0, 122, 204);

            _cellUniverseModel = new CellUniverseModel(_width, _height);
            _cellUniverseModel.OnCellChanged += DrawRect;
        }

        private int GetIntColor(byte red, byte green, byte blue) {
            int result = red << 16;
            result |= green << 8;
            result |= blue << 0;
            return result;
        }

        public void Next() {
            _cellUniverseModel.Next();
        }

        public void ViewRecalculate(int actualWidth, int actualHeight) {

            if (_surfaceWidth != actualWidth || _surfaceHeight != actualHeight) {
                _surfaceWidth = actualWidth;
                _surfaceHeight = actualHeight;
            }
            if (CellUniverseWritableBitmap.Width != _surfaceWidth || CellUniverseWritableBitmap.Height != _surfaceHeight) {
                CellUniverseWritableBitmap = new WriteableBitmap(_surfaceWidth, _surfaceHeight, 96, 96, PixelFormats.Bgr32, null);
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
                    CellUniverseWritableBitmap,
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

                        int pBackBuffer = (int)CellUniverseWritableBitmap.BackBuffer;

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