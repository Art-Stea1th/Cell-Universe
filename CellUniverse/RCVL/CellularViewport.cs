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

namespace RCVL {

    [TemplatePart(Name = CellularViewport.CellSurfaceControl, Type = typeof(Image))]
    public class CellularViewport : Control {

        #region impl. DependencyProperties

        private const string CellSurfaceControl = "PART_CellSurfaceControl";

        private Image  _cellSurfaceControl;

        public static readonly DependencyProperty CellularDataProperty;

        private Color[,] _oldCellularData;
        public Color[,] CellularData {
            get { return (Color[,])GetValue(CellularDataProperty); }
            set { SetValue(CellularDataProperty, value); }
        }

        static CellularViewport() {
            DefaultStyleKeyProperty.OverrideMetadata(
                typeof(CellularViewport), new FrameworkPropertyMetadata(typeof(CellularViewport)));

            CellularDataProperty = DependencyProperty.Register(
                "CellularData", typeof(Color[,]), typeof(CellularViewport), new FrameworkPropertyMetadata(null,
                    FrameworkPropertyMetadataOptions.BindsTwoWayByDefault | FrameworkPropertyMetadataOptions.AffectsRender,
                    new PropertyChangedCallback(OnCellularDataChangedCallback)));
        }

        private static void OnCellularDataChangedCallback(DependencyObject sender, DependencyPropertyChangedEventArgs e) {

            CellularViewport cellViewport = (CellularViewport)sender;
            cellViewport.CellularData     = (Color[,])e.NewValue;

            cellViewport._oldCellularData = (Color[,])e.OldValue;
        }

        #endregion

        private int _surfaceWidth, _surfaceHeight;
        private int _maxCellsHorizontal, _maxCellsVertical;

        private int _spacingBetweenCells = 1;
        private int _cellSize;
        private int _offsetX, _offsetY;

        internal WriteableBitmap CellSurface { get; private set; }

        public override void OnApplyTemplate() {
            base.OnApplyTemplate();
            _cellSurfaceControl = GetTemplateChild(CellSurfaceControl) as Image;
            CorrectParameters();
        }

        private void CorrectParameters() {
            MinWidth  = MinWidth  < 1 ? 1 : MinWidth;
            MinHeight = MinHeight < 1 ? 1 : MinHeight;
        }

        protected override void OnRender(DrawingContext drawingContext) {
            base.OnRender(drawingContext);

            if (CellularData == null) {
                return;
            }

            RecalculateView();
            CellSurface = new WriteableBitmap(_surfaceWidth, _surfaceHeight, 96, 96, PixelFormats.Bgr32, null);

            // --- WPF WriteableBitmap Memory Leak? ---
            GC.Collect();
            // GC.WaitForPendingFinalizers();
            // ----------------------------------------

            _cellSurfaceControl.Source = CellSurface;
            Redraw(CellSurface, _oldCellularData, CellularData);
        }

        #region impl. Recalculate View

        private void RecalculateView() {

            if (_surfaceWidth == (int)ActualWidth && _surfaceHeight == (int)ActualHeight) {
                return;
            }

            _surfaceWidth  = (int)ActualWidth;
            _surfaceHeight = (int)ActualHeight;

            _maxCellsHorizontal = CellularData.GetLength(1);
            _maxCellsVertical = CellularData.GetLength(0);

            _cellSize = GetActualCellSize();

            _offsetX = GetActualOffsetToCenter(
                _surfaceWidth, GetTotalInternalVectorLength(_maxCellsHorizontal, _cellSize, _spacingBetweenCells));

            _offsetY = GetActualOffsetToCenter(
                _surfaceHeight, GetTotalInternalVectorLength(_maxCellsVertical, _cellSize, _spacingBetweenCells));
        }

        private int GetActualCellSize() {
            int maxCellWidth  = GetMaxLengthOfSegmentsInAVector(_maxCellsHorizontal, _surfaceWidth,  _spacingBetweenCells);
            int maxCellHeight = GetMaxLengthOfSegmentsInAVector(_maxCellsVertical,   _surfaceHeight, _spacingBetweenCells);
            return maxCellHeight < maxCellWidth ? maxCellHeight : maxCellWidth;
        }

        private int GetMaxLengthOfSegmentsInAVector(int segmentsCount, int vectorLength, int spacingBetweenSegments) {
            int totalSpacing = (segmentsCount + 1) * spacingBetweenSegments;
            return (vectorLength - totalSpacing) / segmentsCount;
        }

        private int GetActualOffsetToCenter(int externalVectorLenght, int internalVectorLength) {
            return (externalVectorLenght - internalVectorLength) / 2;
        }

        private int GetTotalInternalVectorLength(int segmentsCount, int segmentLength, int spacingBetweenSegments) {
            int totalSegmentsLength = segmentsCount * segmentLength;
            int totalSpacing = (segmentsCount + 1) * spacingBetweenSegments;
            return totalSegmentsLength + totalSpacing;
        }

        #endregion

        #region impl. Redraw

        private void Redraw(WriteableBitmap targetBitmap, Color[,] cellularData) {

            if (targetBitmap == null || cellularData == null) {
                return;
            }

            int width  = cellularData.GetLength(1);
            int height = cellularData.GetLength(0);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    DrawRect(
                        targetBitmap,
                        _offsetX + x * _cellSize + x,
                        _offsetY + y * _cellSize + y,
                        _cellSize, _cellSize,
                        GetIntColor(cellularData[y, x].R, cellularData[y, x].G, cellularData[y, x].B));
                }
            }
        }

        private void Redraw(WriteableBitmap targetBitmap, Color[,] oldCellularData, Color[,] newCellularData) {
            if (targetBitmap == null || oldCellularData == null || newCellularData == null) {
                return;
            }
            foreach (var cell in GetDifference(oldCellularData, newCellularData)) {
                DrawRect(targetBitmap, cell);
            }
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

        private void DrawRect(WriteableBitmap targetBitmap, Tuple<int, int, Color> newState) {
            DrawRect(
                targetBitmap,
                _offsetX + newState.Item1 * _cellSize + newState.Item1,
                _offsetY + newState.Item2 * _cellSize + newState.Item2,
                _cellSize, _cellSize,
                GetIntColor(newState.Item3.R, newState.Item3.G, newState.Item3.B));
        }

        private void DrawRect(WriteableBitmap targetBitmap, int startPosX, int startPosY, int width, int height, int color) {

            if (startPosX + width > (int)ActualWidth || startPosY + height > (int)ActualHeight) {
                return;
            }

            targetBitmap.Lock();
            unsafe
            {
                for (int y = startPosY; y < startPosY + height; ++y) {
                    for (int x = startPosX; x < startPosX + width; ++x) {

                        int pBackBuffer = (int)CellSurface.BackBuffer;

                        pBackBuffer += y * targetBitmap.BackBufferStride;
                        pBackBuffer += x * 4;

                        *((int*)pBackBuffer) = color;
                    }
                }
                targetBitmap.AddDirtyRect(new Int32Rect(startPosX, startPosY, width, height));
            }
            targetBitmap.Unlock();
        }

        private int GetIntColor(byte red, byte green, byte blue) {
            int result = red << 16;
            result |= green << 8;
            result |= blue << 0;
            return result;
        }

        #endregion
    }
}