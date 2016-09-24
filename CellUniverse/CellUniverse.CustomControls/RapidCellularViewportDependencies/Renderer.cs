using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;


namespace CellUniverse.CustomControls.RapidCellularViewportDependencies {

    internal sealed class Renderer {

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
                    if (newCellularData[y, x] == oldCellularData[y, x] && !InvalidateView) {
                        continue;
                    }
                    yield return new Tuple<int, int, Color>(x, y, newCellularData[y, x]);
                }
            }
            InvalidateView = false;
        }

        private void DrawRect(Tuple<int, int, Color> nextRect) {

            int cellIndexX = nextRect.Item1;
            int cellIndexY = nextRect.Item2;
            int cellIColor = GetIntColor(nextRect.Item3.R, nextRect.Item3.G, nextRect.Item3.B);

            int startPosX  = settings.OffsetX + cellIndexX * (settings.CellSize + settings.SpacingBetweenCells);
            int startPosY  = settings.OffsetY + cellIndexY * (settings.CellSize + settings.SpacingBetweenCells);

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
                startPosX + width <= settings.SurfaceWidth &&

                startPosY > 0 &&
                startPosY + height <= settings.SurfaceHeight;
        }
    }
}