using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Resources.Controls {
    using System.Windows.Media.Animation;
    using Helpers;

    public class MatrixLED : FrameworkElement {

        private int cellSize;
        private Size contentSize;
        private Fade fade = new Fade(64);

        private WriteableBitmap mask;
        private WriteableBitmap fadeMask;

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
                false, OnShowFadeChanged));

        static MatrixLED() => StyleProperty.OverrideMetadata(typeof(MatrixLED), new FrameworkPropertyMetadata(CreateStyle()));

        private static Style CreateStyle() {
            var style = new Style(typeof(MatrixLED), null);
            style.Setters.Add(new Setter(FlowDirectionProperty, FlowDirection.LeftToRight));
            style.Seal(); return style;
        }

        private static void OnShowFadeChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) {
            if ((bool)e.NewValue) {
                (d as MatrixLED).fade = new Fade(64);
            }
            else {
                (d as MatrixLED).fade = null;
            }
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

            dc.DrawRectangle(Background, null, new Rect(new Point(), RenderSize));

            if (ShowFade) {

                fade.Add(Source);
                RepaintFadeMask();

                //dc.PushOpacity(0.0, new DoubleAnimation(1.0, 0.0, TimeSpan.FromMilliseconds(1000)).CreateClock());
                dc.PushOpacityMask(new ImageBrush(fadeMask));
                dc.DrawRectangle(Foreground, null, new Rect(new Point(), RenderSize));
                dc.Pop();
                //dc.Pop();
            }            

            RepaintLedsMask();

            dc.PushOpacityMask(new ImageBrush(mask));
            dc.DrawRectangle(Foreground, null, new Rect(new Point(), RenderSize));

        }

        private void RepaintFadeMask() {
            fadeMask = BitmapHelper.Valid(fadeMask, contentSize);
            if (fade != null && fade.Length > 0) {
                using (var context = new WriteableContext(fadeMask)) {

                    if (cellSize == 1) { context.WritePixels(fade); }
                    else { context.WriteCells(fade, cellSize); }
                }
            }
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

        private class Fade {

            private Queue<uint[,]> history;
            private int historySize;

            private uint[,] data;
            public int Length => data.Length;

            public static implicit operator uint[,](Fade fade) => fade.data;

            public Fade(int historySize) {
                this.historySize = historySize > 0 ? historySize : 1;
                history = new Queue<uint[,]>(historySize);
                data = new uint[1, 1];
            }

            public void Add(uint[,] source) {
                if (source == null || source.Length < 1) {
                    Clear();
                    return;
                }
                if (history.Count > 0 &&
                    (history.First().GetLength(0) != source.GetLength(0) ||
                    history.First().GetLength(1) != source.GetLength(1))) {
                    Clear();
                }
                history.Enqueue(source);

                while (history.Count > historySize) {
                    history.Dequeue();
                }
                if (history.Count > 0) {
                    data = Merge(history);
                }
            }

            public void Clear() {
                history = new Queue<uint[,]>(historySize);
                data = new uint[1, 1];
            }

            private uint[,] Merge(IEnumerable<uint[,]> layers) {

                var sumAlpha = new uint[layers.First().GetLength(0), layers.First().GetLength(1)];

                Parallel.ForEach(layers, (layer) => {
                    for (var y = 0; y < layer.GetLength(1); ++y) {
                        for (var x = 0; x < layer.GetLength(0); ++x) {
                            sumAlpha[x, y] += (layer[x, y] >> 24);
                        }
                    }
                });


                var alpha = new uint[sumAlpha.GetLength(0), sumAlpha.GetLength(1)];

                for (var y = 0; y < sumAlpha.GetLength(1); ++y) {
                    for (var x = 0; x < sumAlpha.GetLength(0); ++x) {
                        alpha[x, y] = (sumAlpha[x, y] / (uint)historySize) << 24;
                    }
                }

                return alpha;
            }
        }
    }
}