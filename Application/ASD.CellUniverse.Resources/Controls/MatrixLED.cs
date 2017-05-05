using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Resources.Controls {

    using Helpers;

    public class MatrixLED : FrameworkElement {

        private int cellSize;
        private Size contentSize;
        private WriteableBitmap mask;

        public uint[,] Source { get => (uint[,])GetValue(SourceProperty); set => SetValue(SourceProperty, value); }

        public Brush Background { get => (Brush)GetValue(BackgroundProperty); set => SetValue(BackgroundProperty, value); }
        public Brush Foreground { get => (Brush)GetValue(ForegroundProperty); set => SetValue(ForegroundProperty, value); }


        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(nameof(
            Source), typeof(uint[,]), typeof(MatrixLED), new FrameworkPropertyMetadata(
                null, FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender));

        public static readonly DependencyProperty BackgroundProperty =
            Panel.BackgroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                Brushes.Transparent, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty ForegroundProperty =
            TextElement.ForegroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                SystemColors.ControlTextBrush, FrameworkPropertyMetadataOptions.Inherits | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        static MatrixLED() => StyleProperty.OverrideMetadata(typeof(MatrixLED), new FrameworkPropertyMetadata(CreateStyle()));

        private static Style CreateStyle() {
            var style = new Style(typeof(MatrixLED), null);
            style.Setters.Add(new Setter(FlowDirectionProperty, FlowDirection.LeftToRight));
            style.Seal(); return style;
        }

        protected override Size MeasureOverride(Size constraint) {

            var count = Source == null || Source.Length < 1
                ? (x: 1, y: 1)
                : (x: Source.GetLength(0), y: Source.GetLength(1));

            var scaleFactor = MeasureArrangeHelper.ComputeScaleFactor(constraint, new Size(count.x, count.y));

            cellSize = scaleFactor > 1.0 ? (int)Math.Round(scaleFactor) : 1;
            contentSize = new Size(count.x * cellSize, count.y * cellSize);

            return MeasureArrangeHelper.ComputeSize(constraint, contentSize);
        }

        protected override Size ArrangeOverride(Size arrangeSize) {
            return MeasureArrangeHelper.ComputeSize(arrangeSize, contentSize);
        }

        protected override void OnRender(DrawingContext dc) {

            RepaintLedsMask();

            dc.DrawRectangle(Background, null, new Rect(new Point(), RenderSize));
            dc.PushOpacityMask(new ImageBrush(mask));
            dc.DrawRectangle(Foreground, null, new Rect(new Point(), RenderSize));
        }

        private void RepaintLedsMask() {
            mask = BitmapHelper.Valid(mask, contentSize);
            using (var context = new WriteableContext(mask)) {

                if (cellSize == 1) { context.WritePixels(Source); }
                else { context.WriteCells(Source, cellSize); }
            }
        }
    }
}