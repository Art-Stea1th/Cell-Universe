using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;


namespace CellUniverse.CustomControls.RapidCellularViewportDependencies {

    internal sealed class Renderer { // InverseLogic [Once lock() \ unlock() per frame]

        private Settings settings;
        private WriteableBitmap surface;
        private bool InvalidateView = false;

        internal Renderer(Settings settings) {
            this.settings = settings;
        }

        internal void Update(
            int surfaceWidth, int surfaceHeight, int cellsHorizontal, int cellsVertical, int spacingBetweenCells) {
            if (NeedRecalculateView(surfaceWidth, surfaceHeight, cellsHorizontal, cellsVertical, spacingBetweenCells)) {
                settings.Recalculate(surfaceWidth, surfaceHeight, cellsHorizontal, cellsVertical, spacingBetweenCells);
                surface = new WriteableBitmap(surfaceWidth, surfaceHeight, 96, 96, PixelFormats.Bgr32, null);
                InvalidateView = true;

                // ------- WriteableBitmap Memory Leak? ----------
                // GC.Collect(); // GC.WaitForPendingFinalizers();
                // -----------------------------------------------
            }
        }

        private bool NeedRecalculateView(
            int surfaceWidth, int surfaceHeight, int cellsHorizontal, int cellsVertical, int spacingBetweenCells) {
            return
                settings.SizeChanged(surfaceWidth, surfaceHeight) ||
                settings.CellsCountChanged(cellsHorizontal, cellsVertical) ||
                settings.SpacingBetweenCells != spacingBetweenCells;
        }

        internal unsafe WriteableBitmap Render(Color[,] oldCellularData, Color[,] newCellularData) {

            var s = settings;

            int startX = s.OffsetX;
            int startY = s.OffsetY;
            int width  = (s.CellSize + s.SpacingBetweenCells) * s.CellsHorizontal;
            int height = (s.CellSize + s.SpacingBetweenCells) * s.CellsVertical;

            surface.Lock();

            foreach (var pixel in NextPixel(oldCellularData, newCellularData)) {

                int pBackBuffer = (int)surface.BackBuffer;

                pBackBuffer += pixel.Item2 * surface.BackBufferStride;
                pBackBuffer += pixel.Item1 * 4;

                *((int*)pBackBuffer) = pixel.Item3;
            }

            surface.AddDirtyRect(new Int32Rect(startX, startY, width, height));
            surface.Unlock();

            return surface;
        }

        private IEnumerable<Tuple<int, int, int>> NextPixel(Color[,] oldCellularData, Color[,] newCellularData) {

            if (IsValidData(oldCellularData, newCellularData)) {

                foreach (var cell in GetDifference(oldCellularData, newCellularData)) {

                    int cellIndexX = cell.Item1;
                    int cellIndexY = cell.Item2;
                    int cellIColor = GetIntColor(cell.Item3.R, cell.Item3.G, cell.Item3.B);

                    int startPosX  = settings.OffsetX + cellIndexX * (settings.CellSize + settings.SpacingBetweenCells);
                    int startPosY  = settings.OffsetY + cellIndexY * (settings.CellSize + settings.SpacingBetweenCells);

                    for (int y = startPosY; y < startPosY + settings.CellSize; ++y) {
                        for (int x = startPosX; x < startPosX + settings.CellSize; ++x) {
                            yield return new Tuple<int, int, int>(x, y, cellIColor);
                        }
                    }
                }
            }
        }

        private bool IsValidData(Color[,] oldCellularData, Color[,] newCellularData) {
            return surface != null && oldCellularData != null && newCellularData != null;
        }

        private IEnumerable<Tuple<int, int, Color>> GetDifference(Color[,] oldCellularData, Color[,] newCellularData) {

            int width  = oldCellularData.GetLength(1);
            int height = oldCellularData.GetLength(0);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (newCellularData[y, x] == oldCellularData[y, x] && !InvalidateView) {
                        continue;
                    }
                    yield return new Tuple<int, int, Color>(x, y, newCellularData[y, x]);
                }
            }
            InvalidateView = false;
        }

        private int GetIntColor(byte red, byte green, byte blue) {
            int result = red << 16;
            result |= green << 8;
            result |= blue << 0;
            return result;
        }

        private bool IsValidData(int startPosX, int startPosY, int width, int height) {
            return
                startPosX > 0 &&
                startPosX + width <= settings.SurfaceWidth &&

                startPosY > 0 &&
                startPosY + height <= settings.SurfaceHeight;
        }
    }
}