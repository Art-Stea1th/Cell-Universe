using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace ASD.CellUniverse.Resources.Controls {

    public sealed class AcceptableTextBox : TextBox {

        private string buffer = null;

        public object ToolTipInEdit {
            get => GetValue(ToolTipInEditProperty);
            set => SetValue(ToolTipInEditProperty, value);
        }

        public static readonly DependencyProperty ToolTipInEditProperty = DependencyProperty.Register(
            nameof(ToolTipInEdit), typeof(object), typeof(AcceptableTextBox), new FrameworkPropertyMetadata(new object()));

        public AcceptableTextBox() {
            AcceptsReturn = AcceptsTab = false;
            MinLines = MaxLines = 1;
            Focusable = false;
        }

        static AcceptableTextBox() {
            DefaultStyleKeyProperty.OverrideMetadata(
                typeof(AcceptableTextBox), new FrameworkPropertyMetadata(typeof(AcceptableTextBox)));
        }

        protected override sealed void OnMouseDoubleClick(MouseButtonEventArgs e) {
            buffer = ((AcceptableTextBox)e.Source).Text;
            Focusable = true; Focus(); SelectAll();
        }

        protected override void OnPreviewKeyDown(KeyEventArgs e) {
            if (e.Key == Key.Enter) {
                buffer = ((AcceptableTextBox)e.Source).Text;
            }
            if (e.Key == Key.Enter || e.Key == Key.Escape) {
                MoveFocus(new TraversalRequest(FocusNavigationDirection.Next));
                e.Handled = true;
            }
            base.OnPreviewKeyDown(e);
        }

        protected override sealed void OnLostFocus(RoutedEventArgs e) {
            Text = buffer; Focusable = false; base.OnLostFocus(e);
        }
    }
}