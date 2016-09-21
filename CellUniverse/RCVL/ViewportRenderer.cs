using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;


namespace RCVL {

    internal sealed class ViewportRenderer {

        private ViewportSettings settings;
        private WriteableBitmap   surface;

        internal ViewportRenderer(ViewportSettings settings) {
            this.settings = settings;
        }

        internal void Update(
            int surfaceWidth, int surfaceHeight, int cellsHorizontal, int cellsVertical, int spacingBetweenCells = 1) {
            if (NeedRecalculateView(surfaceWidth, surfaceHeight, cellsHorizontal, cellsVertical, spacingBetweenCells)) {
                settings.Recalculate(surfaceWidth, surfaceHeight, cellsHorizontal, cellsVertical, spacingBetweenCells);
                surface = new WriteableBitmap(surfaceWidth, surfaceHeight, 96, 96, PixelFormats.Bgr32, null);

                // ------- WriteableBitmap Memory Leak? ------->
                GC.Collect(); // GC.WaitForPendingFinalizers();
                // <--------------------------------------------
            }
        }

        private bool NeedRecalculateView(
            int surfaceWidth, int surfaceHeight, int cellsHorizontal, int cellsVertical, int spacingBetweenCells = 1) {
            return
                settings.SizeChanged(surfaceWidth, surfaceHeight) ||
                settings.CellsCountChanged(cellsHorizontal, cellsVertical) ||
                settings.SpacingBetweenCells != spacingBetweenCells;
        }

        internal WriteableBitmap Render(Color[,] oldCellularData, Color[,] newCellularData) {

            if (IsValidData(oldCellularData, newCellularData)) {
                foreach (var cell in GetDifference(oldCellularData, newCellularData)) {
                    DrawRect(cell);
                }
            }
            return surface;
        }

        private bool IsValidData(Color[,] oldCellularData, Color[,] newCellularData) {
            return surface != null && oldCellularData != null && newCellularData != null;
        }

        private IEnumerable<Tuple<int, int, Color>> GetDifference(Color[,] oldCellularData, Color[,] newCellularData) {

            int width  = oldCellularData.GetLength(1);
            int height = oldCellularData.GetLength(0);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (newCellularData[y, x] != oldCellularData[y, x]) {
                        yield return new Tuple<int, int, Color>(x, y, newCellularData[y, x]);
                    }
                }
            }
        }

        private void DrawRect(Tuple<int, int, Color> newState) {

            int cellIndexX = newState.Item1;
            int cellIndexY = newState.Item2;
            int cellIColor = GetIntColor(newState.Item3.R, newState.Item3.G, newState.Item3.B);

            int startPosX  = settings.OffsetX + cellIndexX * settings.CellSize + cellIndexX;
            int startPosY  = settings.OffsetY + cellIndexY * settings.CellSize + cellIndexY;

            DrawRect(startPosX, startPosY, settings.CellSize, settings.CellSize, cellIColor);
        }

        private int GetIntColor(byte red, byte green, byte blue) {
            int result = red << 16;
            result |= green << 8;
            result |= blue << 0;
            return result;
        }

        private void DrawRect(int startPosX, int startPosY, int width, int height, int color) {

            if (IsValidData(startPosX, startPosY, width, height)) {

                surface.Lock();
                unsafe
                {
                    for (int y = startPosY; y < startPosY + height; ++y) {
                        for (int x = startPosX; x < startPosX + width; ++x) {

                            int pBackBuffer = (int)surface.BackBuffer;

                            pBackBuffer += y * surface.BackBufferStride;
                            pBackBuffer += x * 4;

                            *((int*)pBackBuffer) = color;
                        }
                    }
                    surface.AddDirtyRect(new Int32Rect(startPosX, startPosY, width, height));
                }
                surface.Unlock();
            }            
        }

        private bool IsValidData(int startPosX, int startPosY, int width, int height) {
            return
                startPosX > 0 &&
                startPosX + width  <= settings.SurfaceWidth &&

                startPosY > 0 &&                
                startPosY + height <= settings.SurfaceHeight;
        }
    }
}