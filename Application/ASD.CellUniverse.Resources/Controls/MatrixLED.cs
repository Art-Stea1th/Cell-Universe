using System;
using System.Collections.Generic;
using System.Threading.Tasks;
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

        private History history = new History(historySize);
        private const int historySize = 64;

        private WriteableBitmap ledsMask;
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
                (d as MatrixLED).history = new History(historySize);
            }
            else {
                (d as MatrixLED).history = null;
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

                if (history == null) { history = new History(historySize); }

                history.Add(Source);
                RepaintMask(history, ref fadeMask);

                //dc.PushOpacity(0.0, new DoubleAnimation(1.0, 0.0, TimeSpan.FromMilliseconds(1000)).CreateClock());
                dc.PushOpacityMask(new ImageBrush(fadeMask));
                dc.DrawRectangle(Foreground, null, new Rect(new Point(), RenderSize));
                dc.Pop();
                //dc.Pop();
            }

            RepaintMask(Source, ref ledsMask);

            dc.PushOpacityMask(new ImageBrush(ledsMask));
            dc.DrawRectangle(Foreground, null, new Rect(new Point(), RenderSize));

        }

        private void RepaintMask(uint[,] buffer, ref WriteableBitmap mask) {
            mask = BitmapHelper.Valid(mask, contentSize);
            if (buffer != null && buffer.Length > 0) {
                using (var context = new WriteableContext(mask)) {

                    if (cellSize == 1) { context.WritePixels(buffer); }
                    else { context.WriteCells(buffer, cellSize); }
                }
            }
        }

        private class History {

            private int maximumCount;
            private int currentCount;

            private Queue<uint[,]> history;
            private float[,] a/*, r, g, b*/;

            private uint[,] resultData;

            public int Length => resultData.Length;

            public static implicit operator uint[,] (History history) => history.resultData;

            public History(int historyCount) {
                maximumCount = historyCount < 1 ? 1 : historyCount;
                Initialize(1, 1);
            }

            public void Add(uint[,] next) {
                if (next == null || next.Length < 1) {
                    Initialize(1, 1);
                    return;
                }
                if (resultData.GetLength(0) != next.GetLength(0) || resultData.GetLength(1) != next.GetLength(1)) {
                    Initialize(next.GetLength(0), next.GetLength(1));
                }
                Enqueue(next);
                if (currentCount > maximumCount) {
                    Dequeue();
                }
                RecalculateResult();
            }

            public void Initialize(int width, int height) {
                resultData = new uint[width, height];
                history = new Queue<uint[,]>(maximumCount);
                a = new float[width, height];
                //r = new float[width, height];
                //g = new float[width, height];
                //b = new float[width, height];
                currentCount = 0;
            }

            private void Enqueue(uint[,] next) {
                history.Enqueue(next);
                Parallel.For(0, resultData.GetLength(1), (y) => {
                    Parallel.For(0, resultData.GetLength(0), (x) => {
                        a[x, y] += next[x, y] >> 24;
                        //r[x, y] += next[x, y] >> 16 & 0x000000FF;
                        //g[x, y] += next[x, y] >> 8 & 0x000000FF;
                        //b[x, y] += next[x, y] & 0x000000FF;
                    });
                });
                ++currentCount;
            }

            private void Dequeue() {
                var last = history.Dequeue();
                Parallel.For(0, resultData.GetLength(1), (y) => {
                    Parallel.For(0, resultData.GetLength(0), (x) => {
                        a[x, y] -= last[x, y] >> 24;
                        //r[x, y] -= last[x, y] >> 16 & 0x000000FF;
                        //g[x, y] -= last[x, y] >> 8 & 0x000000FF;
                        //b[x, y] -= last[x, y] & 0x000000FF;
                    });
                });
                --currentCount;
            }

            private void RecalculateResult() {
                Parallel.For(0, resultData.GetLength(1), (y) => {
                    Parallel.For(0, resultData.GetLength(0), (x) => {
                        resultData[x, y] =
                        (uint)Limit(a[x, y] / maximumCount) << 24/* |
                        (uint)Limit(r[x, y] / maximumCount) << 16 |
                        (uint)Limit(g[x, y] / maximumCount) << 8 |
                        (uint)Limit(b[x, y] / maximumCount)*/;
                    });
                });
            }

            private byte Limit(float value) => value < 0 ? (byte)0 : value > 255 ? (byte)255 : (byte)value;
        }
    }
}