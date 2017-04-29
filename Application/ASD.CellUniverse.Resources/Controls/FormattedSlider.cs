using System.Reflection;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;

namespace ASD.CellUniverse.Resources.Controls {

    public sealed class FormattedSlider : Slider {

        private ToolTip autoToolTip;

        public string AutoToolTipFormat { get; set; }

        protected override void OnThumbDragStarted(DragStartedEventArgs e) {
            base.OnThumbDragStarted(e);
            FormatAutoToolTipContent();
        }

        protected override void OnThumbDragDelta(DragDeltaEventArgs e) {
            base.OnThumbDragDelta(e);
            FormatAutoToolTipContent();
        }

        private void FormatAutoToolTipContent() {
            if (!string.IsNullOrEmpty(AutoToolTipFormat)) {
                AutoToolTip.Content = string.Format(AutoToolTipFormat, AutoToolTip.Content);
            }
        }

        private ToolTip AutoToolTip {
            get {
                if (autoToolTip == null) {
                    FieldInfo field = typeof(Slider)
                        .GetField("_autoToolTip", BindingFlags.NonPublic | BindingFlags.Instance);
                    autoToolTip = field.GetValue(this) as ToolTip;
                }
                return autoToolTip;
            }
        }
    }
}