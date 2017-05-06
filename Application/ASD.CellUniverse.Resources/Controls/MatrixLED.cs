using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Media.Animation;
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
        public bool ShowFade { get => (bool)GetValue(ShowFadeProperty); set => SetValue(ShowFadeProperty, value); }

        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(nameof(
            Source), typeof(uint[,]), typeof(MatrixLED), new FrameworkPropertyMetadata(
                null, FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender));

        public static readonly DependencyProperty BackgroundProperty =
            Panel.BackgroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                Brushes.Transparent, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty ForegroundProperty =
            TextElement.ForegroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                SystemColors.ControlTextBrush, FrameworkPropertyMetadataOptions.Inherits | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty ShowFadeProperty = DependencyProperty.Register(nameof(
            ShowFade), typeof(bool), typeof(MatrixLED), new FrameworkPropertyMetadata(
                false));

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

        private FadeHelper fh;

        protected override void OnRender(DrawingContext dc) {

            RepaintLedsMask();

            dc.DrawRectangle(Background, null, new Rect(new Point(), RenderSize));

            // bad - code of fade using =)
            if (ShowFade) {
                if (fh == null) {
                    fh = new FadeHelper(1000, 60, Foreground, RenderSize);
                }
                if (fh.RenderSize != RenderSize) {
                    fh.RenderSize = RenderSize;
                }

                fh.Redraw(dc);
                fh.AddImage(mask.Clone());
            }

            dc.PushOpacityMask(new ImageBrush(mask));
            dc.DrawRectangle(Foreground, null, new Rect(new Point(), RenderSize));
        }

        private void RepaintLedsMask() {
            mask = BitmapHelper.Valid(mask, contentSize);
            if (Source != null && Source.Length > 0) {
                using (var context = new WriteableContext(mask)) {

                    if (cellSize == 1) { context.WritePixels(Source); }
                    else { context.WriteCells(Source, cellSize); }
                }
            }
        }

        // bad-impl.fade
        private class FadeHelper {

            private int historySize;
            private Brush brush;
            public Size RenderSize { get; set; }


            private TimeSpan time;
            private Queue<ImageSource> bitmaps;

            private DoubleAnimation fadeAnimation;

            public void AddImage(ImageSource image) {
                bitmaps.Enqueue(image);
                if (bitmaps.Count > historySize) {
                    bitmaps.Dequeue();
                }
            }

            public void Redraw(DrawingContext dc) {

                dc.PushOpacity(0.0, fadeAnimation.CreateClock());

                for (var i = 0; i < bitmaps.Count; ++i) {
                    dc.PushOpacityMask(new ImageBrush(bitmaps.ElementAt(i)));
                    dc.PushOpacity((1.0 / historySize)/* * (i + 1)*/);

                    dc.DrawRectangle(brush, null, new Rect(new Point(), RenderSize));

                    dc.Pop();
                    dc.Pop();
                }
                dc.Pop();
            }

            public FadeHelper(int durationMilliseconds, int historySize, Brush brush, Size renderSize) {

                this.historySize = historySize;
                this.brush = brush;
                this.RenderSize = renderSize;

                // ----

                bitmaps = new Queue<ImageSource>(historySize);

                time = TimeSpan.FromMilliseconds(durationMilliseconds);
                fadeAnimation = new DoubleAnimation(1.0, 0.0, time);
            }
        }
    }
}