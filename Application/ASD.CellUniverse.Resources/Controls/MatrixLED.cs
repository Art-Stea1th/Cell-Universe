using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Resources.Controls {

    using Helpers;

    public class MatrixLED : FrameworkElement {

        private int cellSize = 4, thickness = 1;

        private uint[,] ledsPrev;
        private uint[,] ledsNext;

        private WriteableBitmap gridBitmap, ledsBitmap;

        private ImageBrush GridOpacityMask => new ImageBrush(gridBitmap);
        private ImageBrush LedsOpacityMask => new ImageBrush(ledsBitmap);

        private Size contentSize;

        public uint[,] IntencitySource {
            get => (uint[,])GetValue(IntencitySourceProperty);
            set => SetValue(IntencitySourceProperty, value);
        }

        public Brush Background {
            get => (Brush)GetValue(BackgroundProperty);
            set => SetValue(BackgroundProperty, value);
        }

        public Brush Foreground {
            get => (Brush)GetValue(ForegroundProperty);
            set => SetValue(ForegroundProperty, value);
        }

        public Brush GridBrush {
            get => (Brush)GetValue(GridBrushProperty);
            set => SetValue(GridBrushProperty, value);
        }

        public bool ShowGrid {
            get => (bool)GetValue(ShowGridProperty);
            set => SetValue(ShowGridProperty, value);
        }

        public static readonly DependencyProperty IntencitySourceProperty = DependencyProperty.Register(
            "IntencitySource", typeof(uint[,]), typeof(MatrixLED), new FrameworkPropertyMetadata(
                null, FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender, OnIntencitySourceChanged));

        public static readonly DependencyProperty BackgroundProperty =
            Panel.BackgroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                Brushes.Transparent, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty ForegroundProperty =
            TextElement.ForegroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                SystemColors.ControlTextBrush, FrameworkPropertyMetadataOptions.Inherits | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty GridBrushProperty = DependencyProperty.Register(
            "GridBrush", typeof(Brush), typeof(MatrixLED), new FrameworkPropertyMetadata(
                Brushes.Transparent, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty ShowGridProperty = DependencyProperty.Register(
            "ShowGrid", typeof(bool), typeof(MatrixLED), new FrameworkPropertyMetadata(
                true, FrameworkPropertyMetadataOptions.AffectsRender));

        static MatrixLED()
            => StyleProperty.OverrideMetadata(typeof(MatrixLED), new FrameworkPropertyMetadata(CreateDefaultStyles()));

        private static Style CreateDefaultStyles() {
            var style = new Style(typeof(MatrixLED), null);
            style.Setters.Add(new Setter(FlowDirectionProperty, FlowDirection.LeftToRight));
            style.Seal(); return style;
        }

        private static void OnIntencitySourceChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) {
            var matrix = d as MatrixLED;
            matrix.ledsNext = e.NewValue as uint[,];
            matrix.UpdateBuffer();
            matrix.ledsPrev = matrix.ledsNext;
        }

        private void UpdateBuffer() {

            if (ledsPrev == null) {
                if (ledsNext == null) { return; }
                else { RecalculateMaskSize(); NewGridMask(); NewLedsMask(); }
            }
            else {
                if (ledsNext == null) {
                    ClearLedsMask();
                }
                else {
                    if (SizeEquals(ledsPrev, ledsNext)) { RepaintLedsMask(); }
                    else { RecalculateMaskSize(); NewGridMask(); NewLedsMask(); }
                }
            }
        }

        private static bool SizeEquals<T>(T[,] array1, T[,] array2)
            => array1.GetLength(0) == array2.GetLength(0) && array1.GetLength(1) == array2.GetLength(1);

        private void RecalculateMaskSize() {
            var countX = ledsNext.GetLength(0);
            var countY = ledsNext.GetLength(1);
            var cellsSizeX = countX * cellSize;
            var cellsSizeY = countY * cellSize;
            var thicknessSizeX = countX * thickness + thickness;
            var thisknessSizeY = countY * thickness + thickness;
            contentSize = new Size(cellsSizeX + thicknessSizeX, cellsSizeY + thisknessSizeY);
        }

        private void NewGridMask() {
            gridBitmap = BitmapHelper.CreateWriteable(contentSize);
            RepaintGridMask();
        }

        private void NewLedsMask() {
            ledsBitmap = BitmapHelper.CreateWriteable(contentSize);
            RepaintLedsMask();
        }

        private void RepaintGridMask() {
            using (var context = new WriteableContext(gridBitmap)) {
                context.WriteGrid(cellSize, thickness, (uint)255 << 24);
            }
        }

        private void RepaintLedsMask() {
            using (var context = new WriteableContext(ledsBitmap)) {
                var countX = ledsNext.GetLength(0);
                var countY = ledsNext.GetLength(1);
                for (var x = 0; x < countX; ++x) {
                    for (var y = 0; y < countY; ++y) {
                        var posX = thickness + x * thickness + x * cellSize;
                        var posY = thickness + y * thickness + y * cellSize;
                        context.WriteRect(posX, posY, cellSize, cellSize, ledsNext[x, y]);
                    }
                }
            }
        }

        private void ClearLedsMask() => ledsBitmap = null;

        protected override void OnRender(DrawingContext dc) {

            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.HighQuality);

            dc.DrawRectangle(Background, null, new Rect(new Point(), RenderSize));

            dc.PushOpacityMask(GridOpacityMask);
            dc.DrawRectangle(GridBrush, null, new Rect(new Point(), RenderSize));

            dc.Pop();

            dc.PushOpacityMask(LedsOpacityMask);
            dc.DrawRectangle(Foreground, null, new Rect(new Point(), RenderSize));
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

            if (ledsNext == null) { return; }

            var newCellSize = thickness * 4;

            var purposeWidth = ledsNext.GetLength(0) * newCellSize + ledsNext.GetLength(1) * thickness + 1;
            var availableCellSize = (int)Math.Round(available.Width / purposeWidth) * newCellSize;

            newCellSize = availableCellSize < newCellSize ? newCellSize : availableCellSize;

            if (newCellSize != cellSize) {
                cellSize = newCellSize;
                RecalculateMaskSize();
                NewGridMask();
                NewLedsMask();
            }
        }
    }
}