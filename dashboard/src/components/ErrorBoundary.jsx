import { Component } from "react";

/** Keeps one broken panel from blanking the whole dashboard. */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error(`[${this.props.name ?? "panel"}] render error`, error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="panel">
          <div className="panel-title">{this.props.name ?? "Panel"} failed to render</div>
          <div className="error-text">{String(this.state.error.message ?? this.state.error)}</div>
          <button className="btn" style={{ marginTop: 10 }} onClick={() => this.setState({ error: null })}>
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
