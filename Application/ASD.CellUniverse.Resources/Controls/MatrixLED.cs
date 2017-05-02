using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Resources.Controls {

    using Helpers;

    public class MatrixLED : FrameworkElement {

        private int cellSize, thickness;

        private uint[,] fadeBuffer;
        private uint[,] ledsBuffer;

        private WriteableBitmap fadeBitmap, ledsBitmap, gridBitmap;

        private Size contentSize;

        public uint[,] Matrix { get => (uint[,])GetValue(MatrixProperty); set => SetValue(MatrixProperty, value); }

        public Brush Background {
            get => (Brush)GetValue(BackgroundProperty);
            set => SetValue(BackgroundProperty, value);
        }

        public Brush Foreground {
            get => (Brush)GetValue(ForegroundProperty);
            set => SetValue(ForegroundProperty, value);
        }

        public Brush FadeBrush {
            get => (Brush)GetValue(FadeBrushProperty);
            set => SetValue(FadeBrushProperty, value);
        }

        public Brush GridBrush {
            get => (Brush)GetValue(GridBrushProperty);
            set => SetValue(GridBrushProperty, value);
        }

        public bool ShowGrid {
            get => (bool)GetValue(ShowGridProperty);
            set => SetValue(ShowGridProperty, value);
        }

        public bool ShowFade {
            get => (bool)GetValue(ShowFadeProperty);
            set => SetValue(ShowFadeProperty, value);
        }

        public static readonly DependencyProperty MatrixProperty = DependencyProperty.Register(
            nameof(Matrix), typeof(uint[,]), typeof(MatrixLED), new FrameworkPropertyMetadata(
                null, FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender, OnIntencitySourceChanged));

        public static readonly DependencyProperty BackgroundProperty =
            Panel.BackgroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                Brushes.Transparent, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty ForegroundProperty =
            TextElement.ForegroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                SystemColors.ControlTextBrush, FrameworkPropertyMetadataOptions.Inherits | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty FadeBrushProperty = DependencyProperty.Register(
            nameof(FadeBrush), typeof(Brush), typeof(MatrixLED), new FrameworkPropertyMetadata(
                Brushes.Transparent, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty GridBrushProperty = DependencyProperty.Register(
            nameof(GridBrush), typeof(Brush), typeof(MatrixLED), new FrameworkPropertyMetadata(
                Brushes.Transparent, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty ShowFadeProperty = DependencyProperty.Register(
            nameof(ShowFade), typeof(bool), typeof(MatrixLED), new FrameworkPropertyMetadata(
                false, FrameworkPropertyMetadataOptions.AffectsRender));

        public static readonly DependencyProperty ShowGridProperty = DependencyProperty.Register(
            nameof(ShowGrid), typeof(bool), typeof(MatrixLED), new FrameworkPropertyMetadata(
                false, FrameworkPropertyMetadataOptions.AffectsRender, OnShowGridChanged));

        static MatrixLED()
            => StyleProperty.OverrideMetadata(typeof(MatrixLED), new FrameworkPropertyMetadata(CreateDefaultStyles()));

        private static Style CreateDefaultStyles() {
            var style = new Style(typeof(MatrixLED), null);
            style.Setters.Add(new Setter(FlowDirectionProperty, FlowDirection.LeftToRight));
            style.Seal(); return style;
        }

        private static void OnShowGridChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) {
            //(d as MatrixLED).RepaintGridMask();
        }

        private static void OnIntencitySourceChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) {
            //var matrix = d as MatrixLED;
            //if (e.NewValue is uint[,] newBuffer) {
            //    matrix.ledsBuffer = newBuffer;
            //    matrix.Update();
            //}
            //else {
            //    matrix.Clear();
            //}
        }

        private void Clear() => throw new NotImplementedException();

        private static bool SizeEquals<T>(T[,] array1, T[,] array2)
            => array1.GetLength(0) == array2.GetLength(0) && array1.GetLength(1) == array2.GetLength(1);

        

        private void Update() {
            RecalculateContentSize();
            if (contentSize.Width < 1.0 || contentSize.Height < 1.0) {
                return;
            }
            if (ShowFade) {
                RepaintFadeMask();
            }
            RepaintLedsMask();
            if (ShowGrid) {
                RepaintGridMask();
            }
        }

        private void RecalculateContentSize() {
            var countX = ledsBuffer.GetLength(0);
            var countY = ledsBuffer.GetLength(1);
            var cellsSizeX = countX * cellSize;
            var cellsSizeY = countY * cellSize;
            var thicknessSizeX = countX * thickness + thickness;
            var thisknessSizeY = countY * thickness + thickness;
            contentSize = new Size(cellsSizeX + thicknessSizeX, cellsSizeY + thisknessSizeY);
        }

        private void RepaintFadeMask() {

            if (fadeBitmap == null) {
                fadeBitmap = BitmapHelper.CreateWriteable(contentSize);
            }
            using (var context = new WriteableContext(fadeBitmap)) {
                //...
            }
        }

        private void RepaintLedsMask() {
            if (ledsBitmap == null) {
                ledsBitmap = BitmapHelper.CreateWriteable(contentSize);
            }
            using (var context = new WriteableContext(ledsBitmap)) {
                context.WriteCells(ledsBuffer, cellSize, thickness);
            }
        }

        private void RepaintGridMask() {

            if (gridBitmap == null) {
                gridBitmap = BitmapHelper.CreateWriteable(contentSize);
            }
            using (var context = new WriteableContext(gridBitmap)) {
                context.WriteGrid(cellSize, thickness, (uint)255 << 24);
            }
        }

        private void ClearLedsMask() => ledsBitmap = null;

        protected override void OnRender(DrawingContext dc) {

            //RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.HighQuality);

            dc.DrawRectangle(Background, null, new Rect(new Point(), RenderSize));

            dc.PushOpacityMask(new ImageBrush(ledsBitmap));
            dc.DrawRectangle(Foreground, null, new Rect(new Point(), RenderSize));

            if (ShowGrid) {
                dc.Pop();
                dc.PushOpacityMask(new ImageBrush(gridBitmap));
                dc.DrawRectangle(GridBrush, null, new Rect(new Point(), RenderSize));
            }
        }

        protected override Size MeasureOverride(Size constraint) {
            var resultSize = MeasureArrangeHelper.ComputeSize(constraint, contentSize);
            RecalculateCellSize(resultSize);
            return resultSize;
        }

        protected override Size ArrangeOverride(Size arrangeSize) {
            return MeasureArrangeHelper.ComputeSize(arrangeSize, contentSize);
        }

        private void RecalculateCellSize(Size available) {

            //thickness = 1;

            if (ledsBuffer == null) { return; }

            var newCellSize = thickness * 4;

            var purposeWidth = ledsBuffer.GetLength(0) * newCellSize + ledsBuffer.GetLength(1) * thickness + 1;
            var availableCellSize = (int)Math.Round(available.Width / purposeWidth) * newCellSize;

            newCellSize = availableCellSize < newCellSize ? newCellSize : availableCellSize;

            if (newCellSize != cellSize) {
                cellSize = newCellSize;
                RecalculateContentSize();
            }
        }
    }
}